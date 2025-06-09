import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import psycopg2
import json

# --- Cargar variables de entorno del archivo .env ---
load_dotenv()

# --- Configuración de Flask ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'una_clave_secreta_por_defecto_muy_segura')

# --- Configuración de Google Gemini API ---
genai.configure(api_key=os.getenv('GOOGLE_API_KEY')) 




GOOGLE_CSE_API_KEY = os.getenv('GOOGLE_CSE_API_KEY')
GOOGLE_CSE_CX = os.getenv('GOOGLE_CSE_CX')

# --- Configuración de PostgreSQL ---
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')

def get_db_connection():
    """Establece una conexión a la base de datos PostgreSQL."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None

# --- Rutas y lógica de la aplicación ---

@app.route('/')
def index():
    """Ruta principal que muestra la página de inicio."""
    return render_template('index.html')

@app.route('/iniciar_recoleccion', methods=['POST'])
def iniciar_recoleccion():
    """
    Endpoint para iniciar el proceso de recolección de datos de candidatos
    (esta es la parte donde el "agente" de IA tomará acción).
    """
    print("Iniciando proceso de recolección de datos de candidatos...")
    
    resultado_agente = run_ai_agent_for_data_collection("elecciones presidenciales Colombia 2026")

    return jsonify({"status": resultado_agente})


# --- Funciones de Herramientas (que la IA usará) ---

def google_custom_search(query: str):
    """
    Realiza una búsqueda en Google Custom Search y devuelve una lista de URLs.
    Esta será la "herramienta" de búsqueda web para nuestra IA.
    """
    search_url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_CSE_API_KEY}&cx={GOOGLE_CSE_CX}&q={query}"
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        search_results = response.json()
        
        urls = [item['link'] for item in search_results.get('items', []) if 'link' in item]
        return urls
    except requests.exceptions.RequestException as e:
        print(f"Error al realizar la búsqueda en Google Custom Search: {e}")
        return []
    except Exception as e:
        print(f"Error inesperado en google_custom_search: {e}")
        return []

def scrape_and_extract_text(url: str):
    """
    Descarga el contenido de una URL y extrae el texto principal usando BeautifulSoup.
    Esta será una herramienta para que la IA "lea" la página.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        
        main_content = soup.find('body')
        if main_content:
            for script_or_style in main_content(['script', 'style', 'header', 'footer', 'nav', 'aside']):
                script_or_style.decompose()
            text = main_content.get_text(separator=' ', strip=True)
            return text
        return ""
    except requests.exceptions.RequestException as e:
        print(f"Error al descargar o procesar {url}: {e}")
        return ""
    except Exception as e:
        print(f"Error inesperado en scrape_and_extract_text para {url}: {e}")
        return ""


# --- Función del Agente de IA ---
# --- Función del Agente de IA ---
def run_ai_agent_for_data_collection(target_election: str):
    """
    La función principal del agente de IA para recopilar datos de candidatos.
    Esta función orquestará la búsqueda, scraping y extracción con IA.
    """
    print(f"\n[Agente IA] Iniciando recolección para: {target_election}")

    search_query = f"principales candidatos {target_election} elecciones Colombia"
    print(f"[Agente IA] Decidiendo buscar en la web con la consulta: '{search_query}'")
    
    candidate_urls = google_custom_search(search_query)

    if not candidate_urls:
        print("[Agente IA] No se encontraron URLs relevantes para los candidatos.")
        return "No se encontraron candidatos iniciales."

    print(f"[Agente IA] Encontradas {len(candidate_urls)} URLs para análisis. Procesando...")
    
    collected_candidates_data = []

    for url in candidate_urls[:3]: # Limitar a 3 URLs por ahora para pruebas
        print(f"[Agente IA] Scrapeando y extrayendo texto de: {url}")
        page_text = scrape_and_extract_text(url)

        if not page_text:
            print(f"  [Agente IA] No se pudo extraer texto de {url}. Saltando.")
            continue
        
        # --- NUEVA LÍNEA DE DEPURACIÓN: Mostrar el texto que se le envía a Gemini ---
        print(f"  [Agente IA] Texto extraído para Gemini (primeros 500 chars): {page_text[:500]}...")
        
        # Paso 2: Pedir a la IA (Google Gemini) que extraiga datos estructurados del texto
        print(f"  [Agente IA] Enviando texto a Google Gemini para extracción de datos...")
        extraction_prompt = f"""
        Del siguiente texto sobre un candidato político en Colombia, extrae la siguiente información y devuélvela en formato JSON.
        Si la información no está presente, usa null.

        Información requerida:
        - full_name (Nombre completo del candidato)
        - political_party (Partido político actual o principal)
        - birth_date (Fecha de nacimiento, formato YYYY-MM-DD si es posible)
        - education (Nivel de estudios o títulos universitarios más relevantes)
        - experience (Resumen conciso de experiencia política o laboral clave)
        - main_proposals (Lista de 3 a 5 propuestas principales, cada una como una cadena corta)
        - source_url (La URL de donde se extrajo la información)

        Texto a analizar:
        ---
        {page_text[:4000]} # Limitar el texto para evitar exceder el límite de tokens de la IA
        ---
        DEVUELVE SOLO EL OBJETO JSON Y NADA MÁS. Asegúrate de que la salida sea un JSON válido y completo.
        """
        
        try:
            model = genai.GenerativeModel('models/gemini-1.5-flash-8b')

            response = model.generate_content(
                contents=[{
                    "role": "user",
                    "parts": [{"text": extraction_prompt}]
                }]
            )
            
            # --- NUEVA LÍNEA DE DEPURACIÓN: Mostrar la respuesta cruda de Gemini ---
            print(f"  [Agente IA] Respuesta cruda de Gemini: {response.text}")

            extracted_json_str = response.text.strip().lstrip('```json').rstrip('```')
            
            # --- NUEVO BLOQUE: Intentar parsear el JSON y manejar errores ---
            try:
                # Intentamos convertir la cadena JSON a un objeto Python
                extracted_data = json.loads(extracted_json_str)
                # Si llega aquí, es JSON válido
                print(f"  [Agente IA] Datos extraídos por Gemini (JSON válido): {extracted_data}")
                
                # Puedes guardar el objeto Python directamente si quieres manipularlo
                # Si prefieres seguir con la cadena, no hay problema.
                collected_candidates_data.append(extracted_json_str)

            except json.JSONDecodeError as json_e:
                print(f"  [Agente IA] ERROR: Gemini no devolvió JSON válido. Detalles: {json_e}")
                print(f"  [Agente IA] Cadena JSON intentada: {extracted_json_str}")
                continue # Pasa a la siguiente URL si el JSON es inválido
            except Exception as parse_e:
                print(f"  [Agente IA] ERROR Inesperado al parsear JSON: {parse_e}")
                print(f"  [Agente IA] Cadena JSON intentada: {extracted_json_str}")
                continue


        except Exception as e:
            print(f"  [Agente IA] Error al interactuar con Gemini: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"  [Agente IA] Respuesta cruda de Gemini (si hubo): {response.text}")
            continue

    if collected_candidates_data:
        print("\n[Agente IA] TAREA COMPLETADA: Datos recopilados para los siguientes candidatos:")
        for data in collected_candidates_data:
            # Aquí podríamos parsear data si queremos imprimir el objeto Python, no la cadena
            # print(json.loads(data))
            print(data) # Por ahora, seguimos imprimiendo la cadena JSON
        return "Recolección de datos completada. Revisa la consola para los detalles."
    else:
        return "La recolección de datos no produjo resultados."
# --- Punto de entrada para ejecutar la aplicación Flask ---
if __name__ == '__main__':
    app.run(debug=True)