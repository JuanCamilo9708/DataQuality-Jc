import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import math

from collections import Counter
from ipywidgets import interact
from pandas.api.types import is_numeric_dtype

from Identificacion import *
from Correccion import *

def leer_archivo(archivo, colnames=None, caracter_nan=None):
    
    """
    Función que carga un archivo y devuelve un DataFrame correspondiente.

    Parámetros:
    - archivo: String con el nombre del archivo a cargar.
    - colnames (opcional): Lista con los nombres de las columnas. Si no se proporciona, se usarán los nombres por defecto.
    - caracter_nan (opcional): Carácter que indica un valor vacío en el archivo. Si no se proporciona, no se realizarán cambios.

    Retorna:
    - df: El DataFrame cargado a partir del archivo.

    """
    
    global nombre_archivo  
    nombre_archivo = archivo    
    formato = archivo.split('.')[-1] 
    
    if formato == 'csv':
        df = pd.read_csv(nombre_archivo,names=colnames,na_values=caracter_nan)
    elif formato == 'xlsx' or formato == 'xls':
        df = pd.read_excel(nombre_archivo,names=colnames,na_values=caracter_nan)
    elif formato == 'json':
        df = pd.read_json(nombre_archivo,names=colnames,na_values=caracter_nan)
    else:
        raise ValueError(f"El Formato .{tipo} del archivo no es válido")

    return df


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


def correlacion (df, Umbral = None):
    """
    Función que encuentra las variables más correlacionadas en el DataFrame dado. Retorna solo los pares de variables que superen el umbral.

    Args:
        -df: DataFrame en el que se buscarán las variables correlacionadas.
        -Umbral (float [0-1], opcional): Umbral de correlación. Valor por defecto es 0.5.

    Returns:
        dict: Un diccionario con los pares de variables altamente correlacionadas y su correlación de Pearson.

    """

    matriz_correlacion = df.corr()
    dic_correlacion = {}
    
    if Umbral == None:
        Umbral = 0.5
    for i in range(len(matriz_correlacion.columns)):
        for j in range(i+1, len(matriz_correlacion.columns)):
            cor = matriz_correlacion.iloc[i, j]
            if (abs(cor) > Umbral) or (cor == 1 and (matriz_correlacion.columns[i] != matriz_correlacion.columns[j])):            
                pair = (matriz_correlacion.columns[i], matriz_correlacion.columns[j])            
                dic_correlacion[pair] = cor
    
    sorted_dict = dict(sorted(dic_correlacion.items(), key=lambda x: x[1], reverse=True))


    return(sorted_dict)



def correlacion_nulos (df, Umbral = None):
    
    """
    Encuentra las correlaciones entre variables en el DataFrame dado, teniendo en cuenta los valores nulos. 
    Retorna solo los pares de variables que superen el umbral.

    Args:
        -df (pandas.DataFrame): DataFrame en el que se buscarán las correlaciones.
        -Umbral (float [0-1], opcional): Umbral de correlación. Valor por defecto es 0.5.

    Returns:
        dict: Un diccionario que contiene los pares de variables altamente correlacionadas (por encima del umbral) 
              y su correlación de Pearson, teniendo en cuenta los valores nulos.

    """

    matriz_correlacion = df.isnull().corr()
    dic_correlacion = {}
    
    if Umbral == None:
        Umbral = 0.5
    for i in range(len(matriz_correlacion.columns)):
        for j in range(i+1, len(matriz_correlacion.columns)):
            cor = matriz_correlacion.iloc[i, j]
            if (abs(cor) > Umbral) or (cor == 1 and (matriz_correlacion.columns[i] != matriz_correlacion.columns[j])):            
                pair = (matriz_correlacion.columns[i], matriz_correlacion.columns[j])            
                dic_correlacion[pair] = cor
    
    sorted_dict = dict(sorted(dic_correlacion.items(), key=lambda x: x[1], reverse=True))


    return(sorted_dict)


def grafico_dispersion(df):
    def plotear(x, y):
        try:
            pd.to_numeric(df[x])
            pd.to_numeric(df[y])
            sns.jointplot(x=df[x], y=df[y], kind="hex", joint_kws={"gridsize": 10})
            
        except ValueError:
            print("No es posible graficar las variables seleccionadas, ya que no son numéricas.")

    col_numerica = df.select_dtypes(include=['int64', 'float64']).columns
    interact(plotear, x=col_numerica, y=col_numerica)




def grafica_comparativa (df1, df2, columna):
    
    lista1 = (df1[columna]).dropna()
    lista2 = (df2[columna]).dropna()

    uni1=set(lista1)
    uni2 =set(lista2)
    categorias= (sorted(list(uni1|uni2)))
    
    lab=0
    if len(categorias) > 1 and len(categorias) <20:
        if df1[columna].dtype == 'object':
            categorias1= (sorted(list(set(lista1))))
            valores1 = [categorias1.index(x) for x in lista1]
            valores2 = [categorias1.index(x) for x in lista2]
            lista1 = valores1
            lista2 = valores2
            lab=1
                
                
    if len(categorias) == 2:
        bin=2
    else:
        bin=8
    bins1 = pd.cut(lista1, bins=bin)
    bins2 = pd.cut(lista2, bins=bin)
    bin_edges = bins1.unique().categories
    #bin_edges = bin_edges.map(lambda interval: pd.Interval(round(interval.left, 1), round(interval.right, 1)))
    if bin == 8:
        if lab == 0:
            bin_labels = [str(interval) for interval in bin_edges]  
        else:
            bin_labels = [str(interval) for interval in categorias1]
    else:
        if lab == 0:
            bin_labels = categorias
        else:
            bin_labels = [str(interval) for interval in categorias1]
            
    conteo1 = pd.value_counts(bins1, sort=False)  
    conteo2 = pd.value_counts(bins2, sort=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=bin_labels, y=conteo1.values, name= str(columna+'_Original'), marker_color='blue', marker_opacity=0.7))
    fig.add_trace(go.Bar(x=bin_labels, y=conteo2.values, name=str(columna+'_Imputado'), marker_color='red',marker_opacity=0.8))           
    
    fig.update_layout(
        title=str('Diagrama de barras para '+columna + '_Original / '+ columna + '_Imputado'),
        xaxis_title= columna,
        yaxis_title='Count'
    )
    fig.show()



    