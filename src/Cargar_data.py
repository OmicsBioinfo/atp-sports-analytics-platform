# importamos las librerias necesarias
import pandas as pd 
import glob

'''
Función para obtener cada cargar cada uno de los archivos
'''

def Load_Data(archivos):
    files=glob.glob(archivos)
    Dataframes=[]
    for archivo in files:
        df=pd.read_csv(archivo)
        Dataframes.append(df)
    return Dataframes

'''
Unir los archivos ya establecidos 
'''

def Join_Files(Data):
    Data_ATP=pd.concat(Data, ignore_index=True)
    print("Uniendo los dataframes...")
    print("Dataframe listo. 100%")
    return Data_ATP
