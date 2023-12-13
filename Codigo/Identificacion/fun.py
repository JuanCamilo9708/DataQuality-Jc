import pandas as pd
import numpy as np
import math
import re
import missingno as mi
import seaborn as sns
import plotly.graph_objects as go

from scipy.stats import ks_2samp
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from collections import Counter
from scipy.stats import shapiro
from ipywidgets import interact

from Correccion import *
from Varios import *

import warnings
warnings.filterwarnings("ignore")



def busqueda_valores_ausentes(df, nuevo_valor_nan=None):  
    """
    Función que recibe un DataFrame y devuelve las variables con su caracter que indica nulo, junto con una lista de los 
    posibles caracteres que indican nulos.

    Parámetros:
        - df: DataFrame de pandas.

    Retorna:
        - resultados: Diccionario que mapea cada variable a su posible caracter nulo, indicando que el campo es nulo o tiene un valor ausente.
        - valores_unicos_resultados: Lista de los posibles caracteres nulos encontrados en el DataFrame.

    """
    valores_nan = ['*','-','_','--','---','missing','sinregistro','?','none','nan','',np.nan]
    
    if nuevo_valor_nan != None:
        if type(nuevo_valor_nan) is list:
            for i in nuevo_valor_nan:
                valores_nan.append(i)
        else:
            valores_nan.append(nuevo_valor_nan)  
        
    resultados = {}    
    
    nombre_de_lista_columnas = unicos(df)[0]
    for llave, valores in nombre_de_lista_columnas.items():
        for valor in valores:
            
            if isinstance(valor, str):
                valor_original =valor
                valor = valor.lower().replace(" ", "")   
                
                if valor in valores_nan:
                    if llave not in resultados:
                        resultados[llave] = []
                    resultados[llave].append(valor_original)
            else:
                
                valor = str(valor)       
                    
                if valor in valores_nan:
                    if llave not in resultados:
                        resultados[llave] = []
                    resultados[llave].append(valor)


    valores_unicos_resultados = list(set(valor for lista in resultados.values() for valor in lista))
    
    
    return(resultados,valores_unicos_resultados)


def info_valoresAusentes(df):
    """
    Genera información sobre las variables en un DataFrame en relación con el posible carácter nulo y 
    la cantidad de veces que aparece. Esta función no devuelve ningún valor, solo imprime la información.

    Parámetros:
        - df: DataFrame de pandas.

    """
    
    bva = busqueda_valores_ausentes(df)
    resultados = bva[0]
    valores_unicos_resultados = bva[1]
    
    if len(valores_unicos_resultados) == 1:
        print(f"El DataFrame solo contiene un posible caracter que indica los valores ausentes y es: {valores_unicos_resultados}")
        print()
        
    print('{:<15} {:<25}    {:<25}'.format('Variable', 'Posibles Valores NaN', 'Count'))
    print('-' * 60)
    for k, v in resultados.items():
        df2 = df.fillna('nan')
        conteo = df2[k].value_counts() 
        conteo_filtrado = conteo[v]  
        print('{:<15} {:<35}   {:<35}'.format(k, ', '.join(v), conteo_filtrado.sum()))
        

def Info_objetos_a_numericos(df, umbral=None):
    
    valores_nan = busqueda_valores_ausentes(df)[1]
    uno =['True','S','Si','Yes','Verdadero','V','T']
    cero = ['False','N','No','Falso','F']
    fil = dimensiones_df(df)[0]
    dic={}
    texto = ""
    
    columnas_caso_numericas = []
    lista_column_no_cambiar = []

    columnas_objeto = list(df.select_dtypes('object').columns)
    if len(columnas_objeto) == 0:
        texto +=("No hay variables de tipo Objeto en este DataFrame")
    
        
    if umbral == None:
        umbral= 95

    numeros=0
    strings=0        

    for column in df[columnas_objeto]: 
        for registro in df[column]:
            if registro is None:
                strings  +=1
                continue
            try:   
                if str(registro).capitalize() in uno or str(registro).capitalize() in cero:
                    continue
                if float(registro)/1 >=0:
                    numeros  +=1
            except ValueError:
                if registro.capitalize() not in valores_nan:
                    if column in dic:
                        dic[column].append(registro)
                    else:
                        dic[column] = [registro] 
                        
                    if column not in lista_column_no_cambiar:
                        lista_column_no_cambiar.append(column)                  
                                
                strings  +=1 
                        
        porcentaje_numeros = (numeros/fil) * 100
        porcentaje_strings = (strings/fil) * 100
        
                
        if column not in dic:
            if porcentaje_numeros >= umbral :
                columnas_caso_numericas.append(column)
                texto +=(f"**La Columna '{column}' puede ser numérica debido a:\n")
                texto +=("     Porcentaje de valores numéricos: {:.2f}% ({} de {} de registros)\n".format(porcentaje_numeros, numeros, fil))
                texto +=("     Porcentaje de valores alfabéticos: {:.2f}% ({} de {} de registros)\n\n".format(porcentaje_strings, strings, fil))
                texto += "\n"    
            
        numeros=0
        strings=0
        
    if len(columnas_caso_numericas) == 0:
        texto +=(f"Para el umbral {umbral}%, no hay variables que pueda ser de tipo numerico\n")
    
    if len(columnas_caso_numericas) > 0:
        texto += ("--RESUMEN:--\n")
        texto +=(f"   - Las columnas que cumplieron el umbral de {umbral}%, y pueden ser de tipo numerico:\n")
        texto += str(columnas_caso_numericas)
        texto += "\n"
        
    return(columnas_caso_numericas, texto)


def busqueda_variables_bool_y_categoricas(df):
    """
    Función que identifica las variables booleanas y categóricas en un DataFrame y 
    muestra un resumen de la cantidad de repeticiones de cada dato en una variable categórica.

    Parámetros:
        - df: DataFrame de pandas.
        
    """
    
    lista_NB= df.select_dtypes(exclude='bool').columns.tolist()
    lista_Bnan =[]
    lista_cambioBool = []
    lista_categoria = []
    lista_no_categorica = []
    lista_NB2 = []
    texto = ""
    uno =[1,True,'S','Si','Yes','Verdadero','V','T','True']
    cero = [0,False,'N','No','Falso','F','False']
    
    for i in df[lista_NB]:
        if df[i].dtype == 'object':
            df[i] = df[i].apply(lambda x: str(x).capitalize() if pd.notnull(x) else np.nan)        
            a = list(df[i].unique())
        else:
            a = set(df[i])

        if len(a)<=20:    
            if len(a) == 2: 
                orden = sorted(a)
                zero= False
                one= False
                if orden[0] in cero:            
                    zero=True 
                    if orden[1] in uno:
                        one = True

                if zero == True and one == True:
                    lista_cambioBool.append(i)

            elif len(a) == 3:

                nulo1=False 
                for k in a:
                    if pd.isna(k):
                        nulo1=True
                    else:          
                        lista_sin_nulos = [valor for valor in a if valor is not None]

                if len(lista_sin_nulos) == 2 or nulo1:              

                    lista_Bnan.append(i)
                else:
                    lista_NB2.append(i)
            else:
                lista_NB2.append(i)
                
    if (len(lista_NB2) == 0) :
        texto+=("No hay variables que puedan ser categoricas\n")
    else:
        texto+=("Categoricas\n")
    for j in df[lista_NB2]:
        umbral= df[j].value_counts(dropna=True)
        

                        #Umbral -> la Variable es categorica si hay 20 elementos que se repitan 
        if len(umbral)<=20:
            
            if j not in lista_Bnan:
                lista_categoria.append(j)
            c=0
            count2=0
            texto+=(f"*La Columna {j} es categorica dado a:\n")
            for k in range(len(umbral)):
                elemento=(list(umbral.index)[k])
                count=(list(umbral.values)[k]/len(df[j]))*100
                c+=1               
                if c >5:
                    count2+=count
                    if (c == len(umbral)):            
                        texto+=("   - OTROS constituye el {:.1f}%\n".format(count2))
                else:
                    texto+=("   - {} => {:.1f}%\n".format(elemento,count))



        else:
            texto+=(f"\nLa columna {i} no es categorica")
            lista_no_categorica.append(i)
            
           
    if len(lista_cambioBool)==0:
        texto+=("\n\nNo hay variables que puedan ser booleanas")
    else:
        texto+=("\nLas columnas que pueden ser booleanas son: "+str(lista_cambioBool))
    if len(lista_Bnan)>0:
        texto+=("\nLas columnas que contienen 2 valores, pero no pueden ser de tipo booleana por los nan que tiene: "+str(lista_Bnan))
        
    lista_no_categorica = list(set(lista_NB) - set(lista_cambioBool) - set(lista_Bnan) - set(lista_categoria))

    return(lista_cambioBool, lista_Bnan, lista_categoria, lista_no_categorica, texto)



def outliers(df):
    """
    Identifica los outliers de cada variable en un DataFrame, determinando si sigue o no una distribución normal
    mediante el método Shapiro y aplicando el método de detección de outliers seleccionado (IQR o Zscore).

    Parámetros:
        - df (pandas.DataFrame): El DataFrame que contiene los datos.
 
    Returns:
        - dict: Un diccionario que contiene los resultados de la detección de outliers para cada variable del DataFrame.
      Las llaves del diccionario son los nombres de las variables y los valores son listas que contienen los outliers detectados.

    """
    def outliers_IQR(column):
        Q1 = np.percentile(column, 25)
        Q3 = np.percentile(column, 75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        outliers = column[(column < lower_limit) | (column > upper_limit)]
        return outliers
    
    def outliers_Zscore(column):
        threshold = 3
        z_scores = (column - column.mean()) / column.std()  
        outlier_indices = np.abs(z_scores) > threshold
        outliers = column[outlier_indices]     
        return outliers
    
    
    outliers = pd.DataFrame()
    abc = busqueda_variables_bool_y_categoricas(df)[1]
    columnas_numericas_no_abc = df.select_dtypes(include=['number']).columns.difference(abc)
    
    for col in df[columnas_numericas_no_abc]:
        col2 = list(df[col].dropna())        
        stat, p = shapiro(col2)
        alpha = 0.05
        if p > alpha:
            print(f"\n{col} sigue una distribución normal y es paramétrica y sus outliers son:\n")
            outl = outliers_Zscore(df[col])
            print('{:<5} {:<15}'.format('Fila', 'Valor'))
            print('-' * 15)
            print(outl.to_string(index=True))

        else:
            print(f"{col} no sigue una distribución normal y no es paramétrica y sus outliers son:\n")
            outl = outliers_IQR(df[col])
            print('{:<5} {:<15}'.format('Fila', 'Valor'))
            print('-' * 15)       
            print(outl.to_string(index=True))       

        outliers = pd.concat([outliers, outl], axis=1)
        print("")
    
    return(outliers)


def patrones_columnas(df, Columna=None, Umbral = None):
    """
    Función que identifica los patrones presentes en cada columna de un DataFrame y extrae los registros que no cumplen
    con un umbral mínimo (90% por defecto) de repetición.

    Parámetros:
        - df: DataFrame de pandas.

    """

    def exp_regular(elemento):
        pattern = ""
        for i in elemento.replace(" ",""):
            if i.islower():
                pattern += "a"
            elif i.isupper():
                pattern += "A"
            elif i.isnumeric():
                pattern += "#"
            else:
                er = re.escape(i)
                er = er.replace("\\", "")
                pattern += er
        return pattern
    
    cdq = 0
    list_regex_col=[]
    
    if Columna==None:
        for col in df:
            list_regex_col=[]
            print()
            print("---------------------------------------------------------------")
            print(f"Columna '{col}'")
            print()
            for k in df[col].dropna():
                k = str(k)
                fun=exp_regular(k)
                list_regex_col.append(fun)

            conteo = Counter(list_regex_col)
            ordenado_dic = dict(sorted(conteo.items(), key=lambda x: x[1], reverse=True)) 
            print(f"Patrones: {ordenado_dic}")
            print()
            lista123 = []
            for clave, valor in ordenado_dic.items():
                porcentaje = (valor*100)/df.shape[0]
                if Umbral == None:
                    Umbral = 1
                if porcentaje <= Umbral: # Umbral
                    cl = list(ordenado_dic.keys())
                    lista123.append(clave)
            lista321= []
            for k in df[col].dropna():
                k = str(k)
                cadena=exp_regular(k)

                if cadena in lista123:
                    lista321.append(k)                
            if len(lista321)>0: 
                print()
                print(f"\033[1m Atencion \033[0m En esta columna pueden haber errores debido a que hay registros que no siguen el patrón mayoritario o patrones mayoritarios presentes en la columna.") 
                print(f"Estos registros son:")
                print()
                print(set(lista321))
                print()
    else:
        
        for k in df[Columna].dropna():
            k = str(k)
            fun=exp_regular(k)
            list_regex_col.append(fun)      

        conteo = Counter(list_regex_col)        
        ordenado_dic = dict(sorted(conteo.items(), key=lambda x: x[1], reverse=True)) 
        print(Columna)
        print(f"Patrones: {ordenado_dic}")
        print()        
        lista123 = []
        
        for clave, valor in ordenado_dic.items():
            porcentaje = (valor*100)/df.shape[0]

            if Umbral == None:
                Umbral = 1       
            
            if porcentaje <= Umbral: # Umbral                
                cl = list(ordenado_dic.keys())                 
                lista123.append(clave)
            
        lista321= []
        for k in df[Columna].dropna():
            k = str(k)
            cadena=exp_regular(k)

            if cadena in lista123:
                lista321.append(k) 
                
        if len(lista321)>0: 
            print()
            print(f"\033[1m Atencion \033[0m En esta columna pueden haber errores debido a que hay registros que no siguen el patrón mayoritario o patrones mayoritarios presentes en la columna.") 
            print(f"Estos registros son:")
            print()
            print(set(lista321))
            print()           




def fecha(df):    
    
    def verificar_obj_a_fecha(df):
        columnas_objeto = df.select_dtypes(include=['object']).columns
        columnas_fecha = []
        for columna in columnas_objeto:
            try:
                pd.to_datetime(df[columna])
                columnas_fecha.append(columna)
            except ValueError:
                pass
        return(columnas_fecha)

    def verificar_orden_fechas(fechas):
        fechas = list(fechas)
        if fechas == sorted(fechas):
            return 1
        else:
            return 0
        
    vf = verificar_obj_a_fecha(df)
    if len(vf)!=0:
        for i in vf:
            df[i] = pd.to_datetime(df[i], errors='coerce')
        
    columnas_fecha = df.select_dtypes(include='datetime').columns
    if len(columnas_fecha) ==0:
        print("No hay variables de tipo Fecha")
    else:
        for columna in columnas_fecha:         
            if (verificar_orden_fechas(df[columna])==1):
                diferencia = df[columna].diff()
                patron = diferencia.mode()[0]            
                c=0
                for i, valor in enumerate(diferencia[1:]):
                    if valor != patron :  
                        c+=1                        
                        print(f"- En la columna '{columna}' se observa un patrón de inserción de datos cada '{patron}'. Sin embargo, se identificó un problema entre las filas '{i}' y '{i+1}', donde se encontró una brecha de '{valor}' entre los registros.")
                        print()
                if c == 0 :
                    print(f"La columna '{columna}', no presenta errores en las fechas de las inserciones")
            else:
                print("Fechas arbitrarias")  


def variable_constante(df):
    """    
    Esta función recorre las variables en los datos y verifica si todos sus registros son iguales, 
    lo cual indica que la variable es constante. 
    Retorna una lista que contiene únicamente aquellas variables cuyos registros tienen un único valor.

    Parámetros:
    - df: Un DataFrame de pandas.

    Retorna:
    - list_valores_unicos: Una lista de las variables que solo contienen un valor en sus registros.

    """
    
    list_valores_unicos=[]
    for columna in df:
        if df[columna].dtype == 'object': 
            valores_unicos = list(df[columna].str.strip().str.lower().unique())
            valores_unicos = [valor for valor in valores_unicos if valor is not None and valor != ""]
        else:
            valores_unicos = df[columna].unique()
            valores_unicos = valores_unicos[~np.isnan(valores_unicos)]
            
        if len(valores_unicos)==1:
            list_valores_unicos.append(columna)
    print("Las columnas que solo contienen un valor en sus resgistros son: ")    
    return(list_valores_unicos)

def score_dq(df, pesos=[]):
    # sacar la puntuacion de cantidad de columnas con valores ausentes
    def punt_va(df):
        bva = busqueda_valores_ausentes(df)[0]
        porcentaje = (len(bva) * 100)/ dimensiones_df(df)[1]
        pf= (100 - porcentaje) /100
        return(pf)
    
    #segun la cantidad de nulos que hay en total en el df.
    def punt_nulos(df):
        import sys
        from io import StringIO
        registros_df = dimensiones_df(df)[0] * dimensiones_df(df)[1]
        sys.stdout = StringIO()
        mm = df.copy()
        jj = caracterAusente_a_Nan2(mm)
        nulos= jj.isnull().sum().sum()
        porcentaje_nulos = (nulos * 100)/registros_df
        pf= (100 - porcentaje_nulos) /100
        return(pf)
    
    # sacar la puntuacion de cantidad de columnas con tipo objeto que son numericas
    def punt_on(df):
        ion =Info_objetos_a_numericos(df)[0]
        porcentaje = (len(ion) * 100)/ dimensiones_df(df)[1]
        pf= (100 - porcentaje) /100
        return(pf)
    
    # sacar la puntuacion de cantidad de columnas con tipo bool y categorica 
    def punt_bc(df):
        lista = busqueda_variables_bool_y_categoricas(df)[0] + busqueda_variables_bool_y_categoricas(df)[1] +busqueda_variables_bool_y_categoricas(df)[2]
        suma = len(set(lista))
        porcentaje = (suma * 100)/ dimensiones_df(df)[1]
        pf= (100 - porcentaje) /100
        return(pf)
    
    #para la correlacion 
    def punt_cor(df):
        n = dimensiones_df(df)[1]
        combinaciones = math.comb(n, 2)
        cor = len(correlacion(df,0.8))
        if cor != 0 :
            porcentaje_cor = (cor * 100)/combinaciones
            pf= (100 - porcentaje_cor) /100    
        else:
            pf = 1
        return(pf)
    
    #para la correlacion de nulos
    def punt_cor_nul(df):
        bva = len(busqueda_valores_ausentes(df)[0])
        mm = df.copy()
        n=mm.isnull().any().sum()
        combinaciones = math.comb(n, 2)
        cor = len(correlacion_nulos(mm,0.9))
        if cor != 0 :
            porcentaje_cor = (cor * 100)/combinaciones
            pf= (100 - porcentaje_cor) /100
        else:
            pf = 1
        return(pf)
    
    # para los outliers   
    def punt_outlier(df):
        import sys
        from io import StringIO
        sys.stdout = StringIO()
        d_outli = outliers(df)
        n = dimensiones_df(df)[0]
        n_o = d_outli.shape[0]
        porcentaje_outlier = (n_o * 100)/n
        pf= (100 - porcentaje_outlier) /100
        return(pf)
    
    #verificar la lista de pesos de entrada
    def verificar_lista(lista):
        if len(lista) != 7:
            raise ValueError("La lista debe tener 7 elementos")
        if sum(lista) != 100:
            raise ValueError("La suma de los Pesos no es igual a 100")
        return True
    
    
# 1. cantidad de columnas con valores ausentes
# 2. cantidad de nulos que hay en total en el df
# 3. tipo objeto que son numericas
# 4. tipo bool y categorica 
# 5. correlacion 
# 6. corr nulos  
# 7. outliers

    pf_list = []
    pf_list.append(punt_va(df))
    pf_list.append(punt_nulos(df))
    pf_list.append(punt_on(df))
    pf_list.append(punt_bc(df))
    pf_list.append(punt_cor(df))
    pf_list.append(punt_cor_nul(df))
    pf_list.append(punt_outlier(df))
    
    caracteristicas = ['V.Aus_Col','V.Aus_Df','Obj_Num', 'Bool_Cat', 'Corr', 'Corr_Nulo', 'Outliers']
    
    if pesos == []:
        pesos = [10,35,15,5,10,10,15]
    else:
        if verificar_lista(pesos):
            pesos=pesos
            
        
    
    p_final=[]
    for i in range(7):
        oper = (pf_list[i] * pesos[i])/100
        p_final.append(oper)
        
    dq = pd.DataFrame({'Caracteristicas': caracteristicas, 'Pesos':pesos, 'Puntuacion':pf_list})
    dq.to_csv('dq.csv',index=False)
    
    return(dq, round(sum(p_final),2))