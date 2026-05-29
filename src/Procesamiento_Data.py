import pandas as pd 
import numpy as np

def Clean_Data_ATP(Data_ATP):
    """
    Función para la verificación, limpieza y extracción de componentes de fecha de la ATP.
    """
    df = Data_ATP.copy()
    
    print("==========================================")
    print("  INICIANDO PROCESAMIENTO Y LIMPIEZA ATP  ")
    print("==========================================")
    
    #  1. PROCESAMIENTO DE FECHAS
    print("\n[1/4] Procesando fechas y extrayendo el año...")
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors='coerce')
    
    # CONSERVAR: Extraemos el año directamente de la fecha limpia
    df["Year"] = df["tourney_date"].dt.year
    
    # 2. CORRECCIÓN DE TIPOS DE DATOS 
    print("[2/4] Corrigiendo tipos de datos (IDs y Numéricos)...")
    df['tourney_id'] = df['tourney_id'].astype(str)
    df['winner_id'] = df['winner_id'].astype(str)
    df['loser_id'] = df['loser_id'].astype(str)
    
    df['winner_rank'] = pd.to_numeric(df['winner_rank'], errors='coerce')
    df['loser_rank'] = pd.to_numeric(df['loser_rank'], errors='coerce')
    df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce')

    # 3. LIMPIANDO TEXTO 
    print("[3/4] Limpiando espacios en blanco en columnas de texto...")
    columnas_texto = ['tourney_name', 'surface', 'tourney_level', 'winner_name', 'loser_name', 'winner_hand', 'loser_hand', 'round']
    for col in columnas_texto:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 4. TRATAMIENTO DE NULOS
    print("[4/4] Tratando valores nulos estructurales...")
    columnas_stats = ['w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced',
                      'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced']
    for col in columnas_stats:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    if 'minutes' in df.columns:
        df['minutes'] = df['minutes'].replace(0, np.nan)
        df['minutes'] = df['minutes'].fillna(df['minutes'].median())
    
    print("\n==========================================")
    print("   ¡PROCESAMIENTO FINALIZADO CON ÉXITO!   ")
    print("==========================================")
    
    return df