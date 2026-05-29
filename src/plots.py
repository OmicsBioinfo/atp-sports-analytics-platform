import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def _compute_pca_df(df_source, label_suffix=""):
    """Función interna para calcular el PCA dinámicamente sin duplicar código"""
    pca_cols = ["aces_rate", "df_rate", "service_efficiency", "return_efficiency", "global_win_rate" if "global_win_rate" in df_source.columns else "win_rate"]
    # Limpieza de nulos preventiva
    df_clean = df_source.dropna(subset=pca_cols).copy()
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_clean[pca_cols])
    
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    var_exp = pca.explained_variance_ratio_
    
    df_res = pd.DataFrame(coords, columns=["PC1", "PC2"])
    df_res["Player"] = df_clean["player"].values
    df_res["Win_Rate"] = df_clean[pca_cols[-1]].values
    
    # Marcamos el Top 5 histórico para dejar sus etiquetas siempre fijas
    top_5_legends = ["Novak Djokovic", "Roger Federer", "Rafael Nadal", "Andy Murray", "Andy Roddick", "Carlos Alcaraz","Jannik Sinner"]
    df_res["Show_Label"] = df_res["Player"].apply(lambda x: x if x in top_5_legends else "")
    
    return df_res, var_exp

def Generate_Interactive_EDA_Report(df_global, df_superficies, ai_insights_html, output_path="interfaz.html"):
    print(f"[Plots] Generando componentes analíticos avanzados para {output_path}...")
    template_style = "plotly_white"

    # ---------------------------------------------------------------------------
    # Pestaña 1 del dashboard : EXPLORACIÓN GLOBAL de todas las estadísticas
    fig_global = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Top 15 Jugadores por Win Rate", "Matriz de Correlación Expandida", 
            "Altura vs Tasa de Aces", "Edad vs Tasa de Victoria",
            "Distribución de Dobles Faltas (DF Rate)", "Eficiencia de Servicio vs Devolución"
        ),
        vertical_spacing=0.10, horizontal_spacing=0.10
    )
    
    top_winrate = df_global.sort_values("global_win_rate", ascending=False).head(15)
    fig_global.add_trace(go.Bar(x=top_winrate["global_win_rate"], y=top_winrate["player"], orientation='h', marker=dict(color=top_winrate["global_win_rate"], colorscale="Blues"), showlegend=False), row=1, col=1)
    
    num_cols = ["aces_rate", "df_rate", "service_efficiency", "return_efficiency", "global_win_rate", "avg_age", "avg_height"]
    available_cols = [c for c in num_cols if c in df_global.columns]
    corr_matrix = df_global[available_cols].corr()
    fig_global.add_trace(go.Heatmap(z=corr_matrix.values, x=available_cols, y=available_cols, colorscale="RdBu", zmin=-1, zmax=1, showscale=False), row=1, col=2)
    
    # Dispersiones optimizadas
    fig_global.add_trace(go.Scatter(x=df_global["avg_height"], y=df_global["aces_rate"], mode='markers', hovertemplate="<b>%{text}</b><br>Altura: %{x} cm<br>Aces: %{y:.2%}<extra></extra>", text=df_global["player"], marker=dict(color='#1e3a8a', size=8), showlegend=False), row=2, col=1)
    fig_global.add_trace(go.Scatter(x=df_global["avg_age"], y=df_global["global_win_rate"], mode='markers', hovertemplate="<b>%{text}</b><br>Edad: %{x} años<br>Win Rate: %{y:.2%}<extra></extra>", text=df_global["player"], marker=dict(color='#10b981', size=8), showlegend=False), row=2, col=2)
    fig_global.add_trace(go.Histogram(x=df_global["df_rate"], nbinsx=15, marker_color='#94a3b8', showlegend=False), row=3, col=1)
    
    # Servicio vs Devolución controlado
    fig_global.add_trace(go.Scatter(
        x=df_global["service_efficiency"], y=df_global["return_efficiency"], 
        mode='markers', hovertemplate="<b>%{text}</b><br>Servicio: %{x:.2f}<br>Devolución: %{y:.2f}<extra></extra>",
        text=df_global["player"], marker=dict(size=9, color='#6366f1'), showlegend=False
    ), row=3, col=2)
    
    fig_global.update_layout(height=1100, template=template_style)
    fig_global.update_yaxes(categoryorder="total ascending", row=1, col=1)
    html_raw_global = fig_global.to_html(full_html=False, include_plotlyjs=False)

    # ------------------------------------------------------------------------------------
    # Pestaña 1  (Parte 2): Superficies distintas

    fig_surface = make_subplots(
        rows=2, cols=3,
        specs=[[{"colspan": 1}, {"colspan": 2}, None], [{}, {}, {}]],
        subplot_titles=("Distribución de Aces", "Eficiencia de Devolución", "Top 10 Efectividad en Arcilla (Clay)", "Top 10 Efectividad en Hard", "Top 10 Efectividad en Grass"),
        vertical_spacing=0.15, horizontal_spacing=0.08
    )
    for surf in df_superficies["surface"].unique():
        df_s = df_superficies[df_superficies["surface"] == surf]
        fig_surface.add_trace(go.Box(y=df_s["aces_rate"], name=surf, showlegend=False), row=1, col=1)
        fig_surface.add_trace(go.Box(y=df_s["return_efficiency"], name=surf, showlegend=False), row=1, col=2)

    for i, (s_name, col_color) in enumerate([("Clay", "#ea580c"), ("Hard", "#4f46e5"), ("Grass", "#15803d")], start=1):
        df_f = df_superficies[df_superficies["surface"].str.lower() == s_name.lower()]
        if not df_f.empty:
            top_s = df_f.sort_values("win_rate", ascending=False).head(10)
            fig_surface.add_trace(go.Bar(x=top_s["win_rate"], y=top_s["player"], orientation='h', marker_color=col_color, showlegend=False), row=2, col=i)
            
    fig_surface.update_layout(height=850, template=template_style)
    for c in [1, 2, 3]: fig_surface.update_yaxes(categoryorder="total ascending", row=2, col=c)
    html_raw_surface = fig_surface.to_html(full_html=False, include_plotlyjs=False)

    # ------------------------------------------------------------------------------------------
    # Pestaña 2 del dashboard: MULTI-PCA SEPARADO POR SUPERFICIE Y GLOBAL

    # Calculamos las 4 variaciones del PCA
    df_pca_gl, v_gl = _compute_pca_df(df_global)
    df_pca_cl, v_cl = _compute_pca_df(df_superficies[df_superficies["surface"].str.lower() == "clay"])
    df_pca_hd, v_hd = _compute_pca_df(df_superficies[df_superficies["surface"].str.lower() == "hard"])
    df_pca_gr, v_gr = _compute_pca_df(df_superficies[df_superficies["surface"].str.lower() == "grass"])

    fig_pca = go.Figure()

    # Añadimos las trazas (por defecto solo la primera será visible)
    for df_p, v_exp, visible_status, name in [
        (df_pca_gl, v_gl, True, "Global"), (df_pca_cl, v_cl, False, "Clay"),
        (df_pca_hd, v_hd, False, "Hard"), (df_pca_gr, v_gr, False, "Grass")
    ]:
        fig_pca.add_trace(go.Scatter(
            x=df_p["PC1"], y=df_p["PC2"], mode='markers+text',
            text=df_p["Show_Label"], textposition="top center",
            hovertemplate="<b>%{customdata}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>Win Rate: %{marker.color:.1%}<extra></extra>",
            customdata=df_p["Player"], visible=visible_status, name=name,
            marker=dict(size=10, color=df_p["Win_Rate"], colorscale="Viridis", showscale=True, colorbar=dict(title="Win Rate"), line=dict(width=1, color="DarkSlateGrey"))
        ))

    # Construcción matemática de los botones de interacción de Plotly
    fig_pca.update_layout(
        updatemenus=[dict(
            type="dropdown", direction="down", showactive=True, x=0.02, y=1.15,
            buttons=[
                dict(label="Análisis PCA Global", method="update", args=[{"visible": [True, False, False, False]}, {"xaxis": {"title": f"PC1 ({v_gl[0]*100:.1f}%)"}, "yaxis": {"title": f"PC2 ({v_gl[1]*100:.1f}%)"}}]),
                dict(label="Análisis PCA Arcilla (Clay)", method="update", args=[{"visible": [False, True, False, False]}, {"xaxis": {"title": f"PC1 ({v_cl[0]*100:.1f}%)"}, "yaxis": {"title": f"PC2 ({v_cl[1]*100:.1f}%)"}}]),
                dict(label="Análisis PCA Cemento (Hard)", method="update", args=[{"visible": [False, False, True, False]}, {"xaxis": {"title": f"PC1 ({v_hd[0]*100:.1f}%)"}, "yaxis": {"title": f"PC2 ({v_hd[1]*100:.1f}%)"}}]),
                dict(label="Análisis PCA Césped (Grass)", method="update", args=[{"visible": [False, False, False, True]}, {"xaxis": {"title": f"PC1 ({v_gr[0]*100:.1f}%)"}, "yaxis": {"title": f"PC2 ({v_gr[1]*100:.1f}%)"}}])
            ]
        )],
        template=template_style, height=750,
        xaxis=dict(title=f"Componente Principal 1 ({v_gl[0]*100:.1f}%)"),
        yaxis=dict(title=f"Componente Principal 2 ({v_gl[1]*100:.1f}%)")
    )
    html_raw_pca = fig_pca.to_html(full_html=False, include_plotlyjs=False)

    # --------------------------------------------------------------------------------------------------
    # INYECTAR EN PLANTILLA HTML

    with open(output_path, "r", encoding="utf-8") as f:
        contenido_html = f.read()

    contenido_html = contenido_html.replace("__GRAFICO_GLOBAL__", html_raw_global)
    contenido_html = contenido_html.replace("__GRAFICO_SUPERFICIES__", html_raw_surface)
    contenido_html = contenido_html.replace("__GRAFICO_PCA__", html_raw_pca)
    contenido_html = contenido_html.replace("__AI_INSIGHTS__", ai_insights_html)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(contenido_html)
        
    print(f"🎉 [Plots] ¡Dashboard modular actualizado correctamente!")