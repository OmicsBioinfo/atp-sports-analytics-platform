import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def _prepare_raw_players(Data_ATP):
    """
    Función interna: Procesa el dataset original de partidos, duplica la información
    para Winners y Losers, y calcula todas las métricas derivadas por partido.
    """
    # 1. Columnas a extraer para Ganadores y Perdedores
    cols_w = ["winner_name", "winner_ht", "winner_age", "winner_rank", "winner_rank_points", "minutes",
              "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced",
              "l_svpt", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced", "loser_rank", "surface"]
    
    cols_l = ["loser_name", "loser_ht", "loser_age", "loser_rank", "loser_rank_points", "minutes",
              "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_bpSaved", "l_bpFaced",
              "w_svpt", "w_1stWon", "w_2ndWon", "w_bpSaved", "w_bpFaced", "winner_rank", "surface"]
    
    new_cols = ["player", "height", "age", "rank", "rank_points", "minutes", "aces", "double_faults", 
                "svpt", "first_in", "first_won", "second_won", "bp_saved", "bp_faced",
                "opp_svpt", "opp_first_won", "opp_second_won", "opp_bp_saved", "opp_bp_faced", "opponent_rank", "surface"]

    # 2. Espejar el dataset
    w_df = Data_ATP[cols_w].copy()
    w_df.columns = new_cols
    w_df["win"] = 1
    
    l_df = Data_ATP[cols_l].copy()
    l_df.columns = new_cols
    l_df["win"] = 0
    
    players = pd.concat([w_df, l_df], ignore_index=True)

    # 3. FEATURE ENGINEERING: Métricas de Saque (Service)
    players["aces_rate"] = (players["aces"] / players["svpt"]).fillna(0)
    players["df_rate"] = (players["double_faults"] / players["svpt"]).fillna(0)
    players["first_serve_in_pct"] = (players["first_in"] / players["svpt"]).fillna(0)
    players["first_serve_win_pct"] = (players["first_won"] / players["first_in"]).fillna(0)
    players["second_serve_win_pct"] = (players["second_won"] / (players["svpt"] - players["first_in"])).fillna(0)
    players["bp_save_pct"] = (players["bp_saved"] / players["bp_faced"]).fillna(0)
    players["service_efficiency"] = (players["first_won"] + players["second_won"]) / players["svpt"]
    
    # 4. FEATURE ENGINEERING: Métricas de Devolución (Return)
    players["return_points_won"] = players["opp_svpt"] - (players["opp_first_won"] + players["opp_second_won"])
    players["return_efficiency"] = (players["return_points_won"] / players["opp_svpt"]).fillna(0)
    players["bp_converted"] = players["opp_bp_faced"] - players["opp_bp_saved"]
    players["bp_conversion_pct"] = (players["bp_converted"] / players["opp_bp_faced"]).fillna(0)
    
    # 5. FEATURE ENGINEERING: Dominancia y Contexto
    players["serve_points_lost"] = players["svpt"] - (players["first_won"] + players["second_won"])
    players["dominance_ratio"] = (players["return_points_won"] / players["serve_points_lost"]).fillna(0)
    players["pressure_index"] = (players["bp_save_pct"] * 0.5) + (players["bp_conversion_pct"] * 0.5)
    players["opponent_strength"] = np.where(players["opponent_rank"] > 0, 1 / players["opponent_rank"], 0)
    
    # Limpieza de infinitos matemáticos
    players.replace([np.inf, -np.inf], np.nan, inplace=True)
    return players


def _get_titles_stats(Data_ATP):
    """
    Función interna: Calcula los títulos totales y los puntos de títulos pesados 
    de cada jugador basándose en las rondas finales ('F').
    """
    level_points = {"250": 250, "500": 500, "M": 1000, "G": 2000, "F": 1500, "O": 1200}
    
    # Filtrar solo finales ganadas que pertenezcan a niveles válidos de la ATP
    titles_df = Data_ATP[(Data_ATP["round"] == "F") & (Data_ATP["tourney_level"].isin(level_points.keys()))].copy()
    titles_df["title_points"] = titles_df["tourney_level"].map(level_points)
    
    titles_stats = titles_df.groupby("winner_name").agg(
        weighted_titles=("title_points", "sum"),
        titles=("tourney_id", "count")
    ).reset_index()
    
    titles_stats.rename(columns={"winner_name": "player"}, inplace=True)
    return titles_stats

# FUNCIÓN PRINCIPAL 1: TABLA HISTÓRICA GLOBAL (Para EDA General)

def Build_Global_Player_Features(Data_ATP, min_matches=300):
    """
    Agrupa los datos puramente por JUGADOR. Ideal para análisis descriptivo global.
    Cada fila representa la carrera completa de un tenista único.
    """
    print(f"[Features] Armando Tabla Global (Filtro Carrera >= {min_matches} partidos)...")
    players = _prepare_raw_players(Data_ATP)
    
    # Agrupación global
    global_stats = players.groupby("player").agg({
        "aces_rate": "mean", "df_rate": "mean", "first_serve_in_pct": "mean", "first_serve_win_pct": "mean",
        "second_serve_win_pct": "mean", "bp_save_pct": "mean", "service_efficiency": "mean",
        "return_efficiency": "mean", "bp_conversion_pct": "mean", "dominance_ratio": "mean", "pressure_index": "mean",
        "height": "mean", "age": "mean", "rank": "mean", "rank_points": "mean", "opponent_strength": "mean",
        "minutes": "mean", "win": ["count", "mean"]
    }).reset_index()

    # Aplanar nombres de columnas
    global_stats.columns = ["player", "aces_rate", "df_rate", "first_serve_in_pct", "first_serve_win_pct",
                            "second_serve_win_pct", "bp_save_pct", "service_efficiency", "return_efficiency",
                            "bp_conversion_pct", "dominance_ratio", "pressure_index", "avg_height", "avg_age",
                            "avg_rank", "avg_rank_points", "opponent_strength", "avg_minutes", "total_matches", "global_win_rate"]

    # Aplicar filtro de partidos totales en su carrera
    global_stats = global_stats[global_stats["total_matches"] >= min_matches].reset_index(drop=True)
    
    # Cruzar con la tabla de títulos
    titles = _get_titles_stats(Data_ATP)
    final_df = global_stats.merge(titles, on="player", how="left")
    final_df[["weighted_titles", "titles"]] = final_df[["weighted_titles", "titles"]].fillna(0)
    
    print(f"-> Tabla Global lista con {final_df.shape[0]} jugadores únicos.")
    return final_df


# FUNCIÓN PRINCIPAL 2: TABLA POR SUPERFICIE DE PISTA (Para el PCA y Clústeres)


def Build_Surface_Player_Features(Data_ATP, min_matches_total=300, min_matches_surface=60):
    """
    Agrupa por JUGADOR y SUPERFICIE. Aplica el filtro cruzado inteligente:
    1. El jugador debe ser de élite (>= 300 partidos totales en su carrera).
    2. En la superficie evaluada, debe registrar al menos 60 partidos jugados.
    """
    print(f"[Features] Armando Tabla por Superficie (Global >= {min_matches_total} y Superficie >= {min_matches_surface})...")
    players = _prepare_raw_players(Data_ATP)
    
    # Paso cruzado: obtener la máscara de jugadores que cumplen el mínimo de carrera global
    partidos_totales = players.groupby("player")["win"].count()
    jugadores_elite = partidos_totales[partidos_totales >= min_matches_total].index
    
    # Agrupación por jugador y superficie
    surface_stats = players.groupby(["player", "surface"]).agg({
        "aces_rate": "mean", "df_rate": "mean", "first_serve_in_pct": "mean", "first_serve_win_pct": "mean",
        "second_serve_win_pct": "mean", "bp_save_pct": "mean", "service_efficiency": "mean",
        "return_efficiency": "mean", "bp_conversion_pct": "mean", "dominance_ratio": "mean", "pressure_index": "mean",
        "height": "mean", "age": "mean", "rank": "mean", "rank_points": "mean", "opponent_strength": "mean",
        "minutes": "mean", "win": ["count", "mean"]
    }).reset_index()

    # Aplanar nombres de columnas
    surface_stats.columns = ["player", "surface", "aces_rate", "df_rate", "first_serve_in_pct", "first_serve_win_pct",
                            "second_serve_win_pct", "bp_save_pct", "service_efficiency", "return_efficiency",
                            "bp_conversion_pct", "dominance_ratio", "pressure_index", "avg_height", "avg_age",
                            "avg_rank", "avg_rank_points", "opponent_strength", "avg_minutes", "matches_played", "win_rate"]

    # Aplicación del Doble Filtro Cruzado
    surface_stats = surface_stats[surface_stats["player"].isin(jugadores_elite)]
    surface_stats = surface_stats[surface_stats["matches_played"] >= min_matches_surface].reset_index(drop=True)
    
    # Cruzar con la tabla de títulos
    titles = _get_titles_stats(Data_ATP)
    final_df = surface_stats.merge(titles, on="player", how="left")
    final_df[["weighted_titles", "titles"]] = final_df[["weighted_titles", "titles"]].fillna(0)
    
    print(f"-> Tabla por Superficie lista con {final_df.shape[0]} registros de rendimiento divididos.")
    return final_df