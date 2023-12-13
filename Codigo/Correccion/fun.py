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

from Varios import *
from Identificacion import *


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


def dimensiones_df(df):
    """
    Función que recibe un DataFrame y devuelve el número de filas y columnas.

    Parámetros:
        - df: DataFrame de pandas.

    Retorna:
        - Tupla con el número de filas y columnas del DataFrame.
    """
    fil,col = df.shape
    return(fil,col)

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

def unicos(df):
    """
    Función que recibe un DataFrame y devuelve un diccionario donde la clave es la columna y el valor los valores unicos de la columna.

    Parámetros:
        - df: DataFrame de pandas.

    Retorna:
        - Diccionario con los valores unicos por columna.
        - Texto con la descripción y detalle de los resultados
    """
    texto = ""
    nombre_de_lista_columnas = {}
    for columna in df:
        lista = list((df[columna].value_counts(dropna=False)).index)
        nombre_de_lista_columnas[columna] = lista   
        
   
    for key,value in nombre_de_lista_columnas.items():
        cantidad_valores = len(value)
        texto += str(f"- Variable {key}, tiene {cantidad_valores} valores unicos y son: {value}\n")
        texto += ("\n")
    
    return(nombre_de_lista_columnas , (texto))


def cambiar_objetos_a_numericos(df, umbral=None):
    """
    Función que recibe un DataFrame y un umbral para cambiar las columnas de tipo objeto a numéricas en aquellos casos en los que sea posible.

    Parámetros:
        - df: DataFrame de pandas.
        - umbral (opcional): Umbral por defecto establecido en 95. Indica el porcentaje mínimo de valores numéricos requeridos en una columna para que sea considerada como numérica.

    Retorna:
        - df: DataFrame de pandas modificado.
    """
   
    valores_nan = busqueda_valores_ausentes(df)[1]
    uno =['True','S','Si','Yes','Verdadero','V','T']
    cero = ['False','N','No','Falso','F']
    fil = dimensiones_df(df)[0]
    columnas_caso_numericas = []
    lista_column_no_cambiar = []
    dic={}

    columnas_objeto = list(df.select_dtypes('object').columns)
    if len(columnas_objeto) == 0:
        print("No hay variables de tipo Objeto en este DataFrame")
        
  
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
        numeros=0
        strings=0
        
    if len(columnas_caso_numericas)>0:
        df[columnas_caso_numericas] = df[columnas_caso_numericas].apply(pd.to_numeric, errors='coerce')
    
    if len(columnas_caso_numericas) > 0:
        print(f"**Las columnas que cumplieron el umbral de {umbral}%, y ahora son de tipo numerica:\n")
        print(columnas_caso_numericas)



def caracterAusente_a_Nan2(df, colname=None):
    
    """
    Función que recibe un DataFrame y devuelve el DataFrame modificado con los caracteres ausentes reemplazados por NaN.
    También cambia el tipo de dato de int64 a Int64 y de float64 a Float64 para mejorar el manejo de valores ausentes en el df.

    Parámetros:
        - df: DataFrame. El DataFrame original que se desea modificar.

    Retorna:
        - DataFrame modificado con los caracteres ausentes reemplazados por NaN y los tipos de datos actualizados.
    """
    
    #for i in df.select_dtypes(include='object'):
     #   df[i].replace(np.nan, None, inplace=True)
        
    
    bva=busqueda_valores_ausentes(df)
    lista_caracteres = bva[0]
    variables_a_modificar = list(bva[0].keys())
    cambiar_objetos_a_numericos(df)
    dic_objecto = {}
    dic_numeric = {}
    
    for columna, valores in lista_caracteres.items():
        if df[columna].dtypes == object:
            dic_objecto[columna] = valores
        else:           
            dic_numeric[columna] = valores  
    
 
    for i in dic_objecto.keys():
        for registro in df[i]:
            if registro in dic_objecto[i]:
                df[i].replace(registro, None, inplace=True)
                df[i].replace(np.nan, None, inplace=True)
            
    #if len(dic_numeric)>1:
     #   df = leer_archivo(nombre_archivo, colnames=colname, caracter_nan = dic_numeric)
    print(f"Se modificaron las variables: {variables_a_modificar}")
    print(f"por contener los siguientes caracteres: {lista_caracteres}")   
 
            
    return(df)


def cambiar_variables_bool_y_categoricas(df):
    """
    Función que identifica las variables booleanas y categóricas en un DataFrame, realiza los cambios correspondientes en las 
    variables booleanas y muestra un resumen de la cantidad de repeticiones de cada dato en una variable categórica.

    Parámetros:
        - df: DataFrame de pandas.
        
    """
    
    lista_NB= df.select_dtypes(exclude='bool').columns.tolist()
    lista_Bnan =[]
    lista_cambioBool = []
    lista_categoria = []
    lista_no_categorica = []
    lista_NB2 = []
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
                    df[i].replace(orden[0], 0, inplace=True)
                    df[i].replace(orden[1], 1, inplace=True) 
                    df[i] = df[i].astype(bool)
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

    for j in df[lista_NB2]:
        
        umbral= df[j].value_counts(dropna=False)
        

        #Umbral -> la Variable es categorica si hay 20 elementos que se repitan 
        if len(umbral)<=20:
            if j not in lista_Bnan:
                df[j] = df[j].astype('category')
                lista_categoria.append(j)
		

            
        else:
            lista_no_categorica.append(i)
            
    print("Las columnas que pasaron a ser booleanas son: "+str(lista_cambioBool))
    print("Las columnas que contienen 2 valores, pero no pueden ser de tipo booleana por los nan que tiene: "+str(lista_Bnan))
    print("Las columnas que pasaron a ser categoricas son: "+str(lista_categoria))

    return(lista_cambioBool, lista_Bnan, lista_categoria,lista_no_categorica)


def cor_correos(df):
    """
    Función que corrige posibles errores de escritura en las extensiones de correo en las columnas del DataFrame.

    Esta función analiza las columnas del DataFrame que contienen correos electrónicos y busca posibles
    errores de escritura en las extensiones de correo. Identifica extensiones mal escritas y las corrige.

    Args:
        df: El DataFrame de pandas.

    Returns:
        df: El DataFrame modificado con las extensiones de correo corregidas.
    """
    
    
    def jaccard(string1,string2):   
        """
        Función que calcula la similitud de Jaccard entre dos cadenas.

        Args:
            -str1 (str): La primera cadena.
            -str2 (str): La segunda cadena.

        Returns:
            float: Un valor entre 0 y 1 que representa la similitud de Jaccard.
                   Un valor más cercano a 1 indica mayor similitud, mientras que un valor
                   más cercano a 0 indica menor similitud.
        """
        
        jaccard = len(set(string1).intersection(set(string2))) / len(set(string1).union(set(string2)))
        return jaccard

    
    def identificar_col_correos(df):
        """
        Encuentra las columnas que contienen direcciones de correo electrónico en un DataFrame.

        Args:
            df: El DataFrame en el que se realizará la búsqueda.

        Returns:
            list: Una lista con los nombres de las columnas que contienen correos electrónicos.
        """
        
        columnas_object = df.select_dtypes(include='object').columns
        col_correos= []
        for i in columnas_object:
            verificar_correos = [df[i].iloc[0],df[i].iloc[int(df.shape[0]/2)],df[i].iloc[df.shape[0]-1]]
            c=0
            for registro in verificar_correos:
                if isinstance(registro, (int, float)):
                    continue
                else:                    
                    for caracter in registro:
                        if caracter == '@':
                            c+=1 
            if c==3:
                col_correos.append(i)
        return(col_correos)
    
    
    def corregir_correos_X_columna(df, columna):
        """
        Función que realiza un análisis de similaridad de Jaccard y corrige las extensiones de correo en un DataFrame.

        Esta función analiza la columna de correos contenida en el DataFrame y realiza un análisis de
        similaridad de Jaccard para identificar posibles errores en las extensiones de correo. Si la frecuencia
        de una extensión es menor al 5%, se considera un error y se corrige por la similaridad de mayor frecuencia.
        
        Args:
            df: El DataFrame que contiene la columna de correos a analizar y corregir.
            columna_correos (str): El nombre de la columna de correos a analizar y corregir.

        Returns:
            df: El DataFrame modificado con las extensiones de correo corregidas.
        """

        correos = (df[columna])
        p = []
        for j in correos:
            a=j.split("@")
            p.append(a[1])

        conjunto_correos = set(p)

        dic_frecuencia={}
        contador = Counter(p)
        for elemento, frecuencia in contador.items():
            dic_frecuencia [elemento] = frecuencia
            
            
        dic={}
        umbralcorreo = int((df.shape[0]*5) / 100)
        for i in conjunto_correos:
            ext_i = i.split(".")[0]
            for j in conjunto_correos:
                ext_j = j.split(".")[0]
                if not ext_i == ext_j and jaccard(ext_i,ext_j)>0.7 :
                    #  menos del 5% de los registros se considera error
                    if (dic_frecuencia[i] > umbralcorreo or dic_frecuencia[j] > umbralcorreo):
                        if (dic_frecuencia[i] > dic_frecuencia[j]):
                            if i in dic:
                                if isinstance(dic[i], list):
                                     if j not in dic[i]:
                                        dic[i].append(j)
                                else:
                                    dic[i] = [dic[i], j]
                            else:
                                dic[i] = j
                        else:
                            if j in dic:
                                if isinstance(dic[j], list):
                                     if i not in dic[j]:
                                        dic[j].append(i)
                                else:
                                    dic[j] = [dic[j], i]
                            else:
                                dic[j] = i
        if len(dic) == 0:
            print("No hay errores en las extensiones de los correos")
        else:
            for clave, valores in dic.items():
                print(f"se cambió {valores} por {clave}")
            print()

        lista_valores = list(set(valor for sublist in dic.values() for valor in sublist))
        # Correcion correos
        cont = 0
        for c in df[columna]:
            partes = c.split('@')
            user = partes[0]
            dominio = partes[1]
            if dominio in lista_valores:               
                for clave, valores in dic.items():   
                    if dominio in valores:
                        #df[columna][cont] = str(user +'@'+ clave)
                        df.loc[cont,columna] = str(user + '@' + clave)
            cont+=1

        return (df)


    
    col_correos= identificar_col_correos(df)
    if len(col_correos)==0:
        print("En el dataset, no se encontraron columnas con correos")
        return()
    else:
        for i in col_correos:
            print(f"*En la columna {i}:")
            corregir_correos_X_columna(df,i)
    return(df)


def ks_pvalue(df1, df2, col):
    column1 = df1[col]
    column2 = df2[col]

    _, p_value = ks_2samp(column1, column2)
    return p_value

def ks (df1, df2, col):
    datos1 = df1[col].dropna()
    datos2 = df2[col].dropna()
    uni1=set(datos1)
    uni2 =set(datos2)
    categorias= (sorted(list(uni1|uni2)))
   
    valores1 = [categorias.index(x) for x in datos1]
  

    valores2 = [categorias.index(x) for x in datos2]
    _, p_value = ks_2samp(valores1, valores2)
    
    return p_value

def diff_std(df1, df2, col):
    diff = df1[col].std() - df2[col].std()
    return(abs(diff))


def diff_corr(df1, df2):
    correlacion_original = df1.corr()
    correlacion_imputado = df2.corr()

    diferencias_correlacion = correlacion_imputado - correlacion_original   
    dic_cambio_corr = {}

    
    for i in range(len(diferencias_correlacion.columns)):
        for j in range(i+1, len(diferencias_correlacion.columns)):
            cambio_correlacion = diferencias_correlacion.iloc[i, j]
            
            if abs(cambio_correlacion) >= 0.2:
                variable_1 = diferencias_correlacion.columns[i]
                variable_2 = diferencias_correlacion.columns[j]
                dic_cambio_corr[(variable_1, variable_2)] = cambio_correlacion

   
    #for par, cambio in dic_cambio_corr.items():
     #   print(f"{par}: {cambio}")

    return(len(dic_cambio_corr))

   
    
def seleccionar_tipo_imputacion(lista_ks, lista_std, lista_cor):
    #return 0 -> mean, 1 -> median, 2 -> knn
    
    len_ks =len(set(lista_ks))
    len_std = len(set(lista_std))
    len_cor = len(set(lista_cor))
    
    
    if len_ks == 1 and len_std == 1 and len_cor == 1:
        return random.randint(0, 2)

    valor_max_ks = max(lista_ks)
    if valor_max_ks > 0.05:
        posicion_ks = lista_ks.index(valor_max_ks) 
        return posicion_ks 
    else:         
        if len_std == 1 and len_cor>1:
            valor_min_cor = min(lista_cor)
            pos_cor = lista_cor.index(valor_min_cor) 
            return pos_cor
        else:
            valor_min_std = min(lista_std)
            pos_std = lista_std.index(valor_min_std) 
            return pos_std     


    
def imputacion2 (df):
    
    boN = busqueda_variables_bool_y_categoricas(df)[1]
    cate = busqueda_variables_bool_y_categoricas(df)[2]
    Nc = busqueda_variables_bool_y_categoricas(df)[3]    
    cate2 = boN + cate

    df_imp = df.copy()


    #Bool, Cate
    if len(cate2)!=0:
        for col in cate2:
            if df[col].dtype != 'object' or not pd.api.types.is_datetime64_any_dtype(df[col]):
                if (df[col].isnull().sum())>0: 
                    si = SimpleImputer(strategy='most_frequent')
                    df_imp[col] = si.fit_transform(df_imp[col].values.reshape(-1,1))
    
    #No Cate
    if len(Nc)!=0:    
        for col in Nc:            
           
            if pd.api.types.is_datetime64_any_dtype(df[col]) == False:
                lista_KS=[]
                lista_std=[]
                lista_cor=[]

                if (df[col].isnull().sum())>0:
                    if df[col].dtype == 'object' :                        
                        si = SimpleImputer(strategy='most_frequent')  
                        df_imp[col] = si.fit_transform(df_imp[col].values.reshape(-1,1))
                            
                    else:
                        for i in range(3):
                            if i ==0:
                                si = SimpleImputer(strategy='mean')  
                            if i == 1 :
                                si = SimpleImputer(strategy='median')                
                            if i ==2:
                                si = KNNImputer(n_neighbors=3, weights="uniform")


                            df_imp2 = df.copy()
                            df_imp2[col] = si.fit_transform(df_imp2[col].values.reshape(-1,1))

                            valor= ks_pvalue (df, df_imp2, col)
                            lista_KS.append(valor)

                            diff=diff_std(df, df_imp2, col)
                            lista_std.append(diff)

                            dc=diff_corr(df, df_imp2)
                            lista_cor.append(dc)


                if len(lista_KS) != 0: 
                    select = seleccionar_tipo_imputacion(lista_KS, lista_std, lista_cor)

                    if select ==0:
                        si = SimpleImputer(strategy='mean')   
                        df_imp[col] = si.fit_transform(df_imp[col].values.reshape(-1,1))
                    elif select == 1 :
                        si = SimpleImputer(strategy='median')  
                        df_imp[col] = si.fit_transform(df_imp[col].values.reshape(-1,1))
                    else:
                        si = KNNImputer(n_neighbors=3, weights="uniform")
                        df_imp[col] = si.fit_transform(df_imp[col].values.reshape(-1,1))

                    
            else:
                print(f"la {col} no se imputo ya que es tipo Fecha ")
    
    return df_imp