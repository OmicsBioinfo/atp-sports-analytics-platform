from src.Cargar_data import Load_Data, Join_Files
from src.Procesamiento_Data import Clean_Data_ATP
from src.Features import Build_Global_Player_Features, Build_Surface_Player_Features
from src.plots import Generate_Interactive_EDA_Report
import os 
import webbrowser
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------------------------------
# 1. PIPELINE para extraer la data, limpiarla y construir métricas 
atp_1 = Load_Data("data/raw/20*.csv")
atp_2 = Join_Files(atp_1)
atp_3 = Clean_Data_ATP(atp_2)
atp_4 = Build_Global_Player_Features(atp_3)
atp_5 = Build_Surface_Player_Features(atp_3)

# =========================================================================
# 2. Uso de IA aplicada: GROQ CLOUD 

def analyze_eda_automated(df_global):
    print("[AI Engine] Enviando matrices multivariantes a Groq Cloud...")
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    stats_summary = df_global.describe().to_string()
    
    # Agregamos contexto explícito de los resultados matemáticos que me pasaste
    multivariant_context = """
    Resultados del Modelo PCA ejecutado:
    - ALL SURFACES: Varianza Explicada ~66.1% (PC1: 48.2%, PC2: 17.9%). Atributos dominantes en PC1: dominance_ratio, win_rate, avg_rank_points. Atributos en PC2: return_efficiency vs aces_rate.
    - CLAY: Varianza Explicada ~60.9%. PC2 está fuertemente dominado de forma inversa por return_efficiency (-0.38) y aces_rate (0.46). Nadal lidera con score de dominancia de 19.3.
    - HARD: Varianza Explicada ~63.8%. Djokovic domina la superficie (score 21.5) seguido de cerca por Federer (19.7).
    - GRASS: Varianza Explicada ~71.0% (La más alta). PC2 dominado por aces_rate (0.44) y service_efficiency (0.38). Federer tiene una dominancia histórica de 10.58, superando ampliamente a Djokovic (6.46).
    """

    prompt = f"""
    Eres un científico de datos senior y experto en Sports Analytics del circuito ATP.
    Analiza con máximo rigor estadístico el siguiente resumen y los resultados del PCA por superficie:

    {multivariant_context}

    ======================================
    RESUMEN ESTADÍSTICO GENERAL
    ======================================
    {stats_summary}

    Explica de forma detallada y profesional en un reporte científico de 11 puntos:
    1. Qué patrones biométricos y de juego observas.
    2. Qué variables determinan el éxito según los componentes del PCA.
    3. Interpretación táctica de 'service_efficiency' y 'aces_rate' en Grass vs Clay.
    4. Compara a Roger Federer, Novak Djokovic y Rafael Nadal usando las métricas de dominancia obtenidas en el PCA y determina de forma matemática quién es el jugador más completo en base a los datos.

    CRÍTICO: Devuelve la respuesta estructurada directamente con etiquetas HTML limpias (<h3>, <p>, <strong>, <ul>, <li>). No pongas bloques de código ni texto plano.
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "Eres un científico de datos senior experto en tenis que genera reportes profesionales estructurados nativamente en HTML."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )
    return response.choices[0].message.content

# Ejecución del motor conversacional
reporte_ia_html = analyze_eda_automated(atp_4)

# ----------------------------------------------------------------------------
# 3. COMPILACIÓN E INYECCIÓN EN DASHBOARD

archivo_interfaz = "interfaz.html"
Generate_Interactive_EDA_Report(atp_4, atp_5, reporte_ia_html, output_path=archivo_interfaz)

print("[Pipeline] Dashboard unificado completado con éxito.")
webbrowser.open('file://' + os.path.realpath(archivo_interfaz))