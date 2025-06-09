import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import psycopg2
import json
from google.api_core.exceptions import ResourceExhausted

# Cargar variables de entorno
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_secret_key')

# --- Configuración de Google Gemini API ---
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# --- Función de Google Custom Search ---
def google_custom_search(query, num_results=5): # Añadimos num_results para mayor control
    api_key = os.getenv("CUSTOM_SEARCH_API_KEY")
    cx_id = os.getenv("CUSTOM_SEARCH_ENGINE_ID")

    if not api_key or not cx_id:
        print("Error: CUSTOM_SEARCH_API_KEY o CUSTOM_SEARCH_ENGINE_ID no configurados en .env")
        return []

    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx_id}&q={query}&num={num_results}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        search_results = response.json()
        
        urls = [item['link'] for item in search_results.get('items', [])]
        return urls
    except requests.exceptions.RequestException as e:
        print(f"Error en la búsqueda de Google Custom Search: {e}")
        return []

# --- Función de Web Scraping ---
def scrape_and_extract_text(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() 
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.extract()
        
        text = soup.get_text()
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error al hacer scraping de {url}: {e}")
        return None

# --- Función para enriquecer datos de un candidato ---
def enrich_candidate_data(candidate_info: dict, candidate_name: str, initial_source_url: str):
    """
    Intenta buscar información faltante para un candidato específico.
    """
    print(f"    [Agente IA - Enriquecimiento] Intentando enriquecer datos para: {candidate_name}")
    
    # Campos a verificar si falta información
    fields_to_check = ['birth_date', 'education', 'experience', 'main_proposals']
    
    # Almacenar la URL original si no está presente
    if not candidate_info.get('source_url'):
        candidate_info['source_url'] = initial_source_url

    # Bucle para unas pocas búsquedas adicionales por candidato
    max_enrichment_searches = 2 # Límite de búsquedas extra por campo para evitar un uso excesivo de la API

    for field in fields_to_check:
        # Verificar si el campo está vacío, es 'null' o una lista vacía/solo con 'null'
        is_missing = False
        if candidate_info.get(field) is None or candidate_info.get(field) == 'null':
            is_missing = True
        elif field == 'main_proposals' and (not candidate_info.get(field) or candidate_info.get(field) == ['null'] or candidate_info.get(field) == []):
            is_missing = True
        
        if is_missing:
            print(f"      [Agente IA - Enriquecimiento] Buscando información faltante para '{field}' de {candidate_name}")
            
            query = ""
            if field == 'birth_date':
                query = f"Fecha de nacimiento de {candidate_name}"
            elif field == 'education':
                query = f"Estudios y educación de {candidate_name}"
            elif field == 'experience':
                query = f"Experiencia política de {candidate_name} Colombia"
            elif field == 'main_proposals':
                query = f"Propuestas principales de {candidate_name} elecciones presidenciales Colombia"
            
            if not query:
                continue

            enrichment_urls = google_custom_search(query)
            
            urls_for_field = 0
            for enr_url in enrichment_urls:
                if urls_for_field >= max_enrichment_searches:
                    break # Detener después de unas pocas URLs para este campo
                if enr_url == initial_source_url: # No volver a raspar la página inicial
                    continue
                
                print(f"        [Agente IA - Enriquecimiento] Scrapeando URL adicional para {field}: {enr_url}")
                enr_page_text = scrape_and_extract_text(enr_url)
                
                if enr_page_text:
                    enrichment_prompt = f"""
                    Del siguiente texto sobre {candidate_name}, extrae específicamente la información para el campo '{field}'.
                    Si la información no está presente, usa null (o una lista vacía para 'main_proposals').
                    DEVUELVE SOLO UN OBJETO JSON con la clave '{field}' y su valor.

                    Texto a analizar:
                    ---
                    {enr_page_text[:3000]}
                    ---
                    """
                    try:
                        model = genai.GenerativeModel('models/gemini-1.5-flash')
                        response = model.generate_content(
                            contents=[{
                                "role": "user",
                                "parts": [{"text": enrichment_prompt}]
                            }]
                        )
                        enr_json_str = response.text.strip().lstrip('```json').rstrip('```')
                        
                        try:
                            enr_data = json.loads(enr_json_str)
                            # Actualizar el campo si el valor extraído no es None/null/lista vacía (para propuestas)
                            if field in enr_data and (enr_data[field] is not None and (field != 'main_proposals' or (field == 'main_proposals' and enr_data[field] != ['null'] and enr_data[field] != []))):
                                candidate_info[field] = enr_data[field]
                                print(f"          [Agente IA - Enriquecimiento] '{field}' encontrado y actualizado.")
                                break # Se encontró la información para este campo, pasar al siguiente campo faltante
                            else:
                                print(f"          [Agente IA - Enriquecimiento] '{field}' no encontrado en esta URL o valor no útil.")
                        except json.JSONDecodeError:
                            print(f"          [Agente IA - Enriquecimiento] Gemini no devolvió JSON válido para '{field}'.")
                        except Exception as e:
                            print(f"          [Agente IA - Enriquecimiento] Error al procesar JSON para '{field}': {e}")
                    except ResourceExhausted as re:
                        print(f"        [Agente IA - Enriquecimiento] Error de Cuota/Recurso durante enriquecimiento: {re}")
                        break # Detener el enriquecimiento de este candidato si se alcanza la cuota
                    except Exception as e:
                        print(f"        [Agente IA - Enriquecimiento] Error al interactuar con Gemini para enriquecimiento: {e}")
                urls_for_field += 1
    
    print(f"    [Agente IA - Enriquecimiento] Enriquecimiento completado para: {candidate_name}. Datos finales: {candidate_info}")
    return candidate_info

# --- Función del Agente de IA ---
def run_ai_agent_for_data_collection(target_election: str):
    print(f"\n[Agente IA] Iniciando recolección para: {target_election}")

    search_query = f"principales candidatos {target_election} elecciones Colombia"
    print(f"[Agente IA] Decidiendo buscar en la web con la consulta: '{search_query}'")
    
    candidate_urls = google_custom_search(search_query)

    if not candidate_urls:
        print("[Agente IA] No se encontraron URLs relevantes para los candidatos.")
        return [], "No se encontraron candidatos iniciales."

    print(f"[Agente IA] Encontradas {len(candidate_urls)} URLs para análisis. Procesando...")
    
    collected_candidates_data = []
    collected_party_names = set() # Para rastrear partidos ya recopilados
    collected_candidate_names = set() # Para rastrear nombres de candidatos ya recopilados

    # Configuración de límites para la recolección
    max_initial_urls_to_check = 15 # Cuántas de las URLs encontradas en la búsqueda intentaremos procesar inicialmente
    target_unique_candidates = 5 # Cuántos candidatos únicos (por partido o nombre) queremos recopilar
                               # ¡CAMBIO AQUÍ: de 3 a 5!

    urls_checked_count = 0
    current_url_index = 0

    # Bucle para encontrar candidatos únicos hasta alcanzar el objetivo o quedarse sin URLs
    while len(collected_candidates_data) < target_unique_candidates and current_url_index < len(candidate_urls) and urls_checked_count < max_initial_urls_to_check:
        url = candidate_urls[current_url_index]
        current_url_index += 1
        urls_checked_count += 1

        print(f"\n[Agente IA] Procesando URL inicial: {url} (URL {urls_checked_count}/{max_initial_urls_to_check})")
        page_text = scrape_and_extract_text(url)

        if not page_text:
            print(f"  [Agente IA] No se pudo extraer texto de {url}. Saltando a la siguiente URL.")
            continue
        
        print(f"  [Agente IA] Texto extraído para Gemini (primeros 500 chars): {page_text[:500]}...")
        
        print(f"  [Agente IA] Enviando texto a Google Gemini para extracción de datos inicial...")
        # Prompt para la extracción inicial de TODOS los campos
        initial_extraction_prompt = f"""
        Del siguiente texto sobre un candidato político en Colombia, extrae la siguiente información y devuélvela en formato JSON.
        Si la información no está presente, usa null.

        Información requerida:
        - full_name (Nombre completo del candidato)
        - political_party (Partido político actual o principal)
        - birth_date (Fecha de nacimiento, formato YYYY-MM-DD si es posible)
        - education (Nivel de estudios o títulos universitarios más relevantes)
        - experience (Resumen conciso de experiencia política o laboral clave)
        - main_proposals (Lista de 3 a 5 propuestas principales, cada una como una cadena corta. Si no hay propuestas, usa una lista vacía `[]`)
        - source_url (La URL de donde se extrajo la información)

        Texto a analizar:
        ---
        {page_text[:4000]}
        ---
        DEVUELVE SOLO EL OBJETO JSON Y NADA MÁS. Asegúrate de que la salida sea un JSON válido y completo.
        """
        
        try:
            model = genai.GenerativeModel('models/gemini-2.0-flash-lite')

            response = model.generate_content(
                contents=[{
                    "role": "user",
                    "parts": [{"text": initial_extraction_prompt}]
                }]
            )
            
            extracted_json_str = response.text.strip()
            if extracted_json_str.startswith('```json'):
                extracted_json_str = extracted_json_str.lstrip('```json').rstrip('```')
            
            print(f"  [Agente IA] Respuesta cruda de Gemini (inicial): {extracted_json_str[:500]}...")

            try:
                extracted_data = json.loads(extracted_json_str)
                
                candidate_name = extracted_data.get('full_name')
                party_name = extracted_data.get('political_party')

                # Normalizar nombres para comparación
                normalized_candidate_name = candidate_name.strip().lower() if candidate_name else None
                normalized_party_name = party_name.strip().lower() if party_name else None

                is_unique = True

                # Verificar si el resultado es útil y único
                if not normalized_candidate_name:
                    print(f"  [Agente IA] ERROR: No se extrajo un 'full_name' útil de {url}. Saltando a la siguiente URL.")
                    is_unique = False
                elif normalized_candidate_name in collected_candidate_names:
                    print(f"  [Agente IA] Candidato '{candidate_name}' ya recopilado. Saltando a la siguiente URL.")
                    is_unique = False
                elif normalized_party_name and normalized_party_name in collected_party_names:
                    print(f"  [Agente IA] Partido '{party_name}' ya recopilado. Saltando a la siguiente URL.")
                    is_unique = False
                
                if is_unique:
                    # Si es un candidato/partido nuevo y útil, se procede a enriquecerlo
                    print(f"  [Agente IA] Candidato/Partido nuevo detectado: '{candidate_name}' de '{party_name}'. Iniciando enriquecimiento.")
                    
                    # Llamar a la función para enriquecer los datos
                    enriched_data = enrich_candidate_data(extracted_data, candidate_name, url)
                    
                    collected_candidates_data.append(enriched_data)
                    collected_candidate_names.add(normalized_candidate_name)
                    if normalized_party_name:
                        collected_party_names.add(normalized_party_name)
                else:
                    pass # No es único o útil, se saltará y se continuará el bucle while

            except json.JSONDecodeError as json_e:
                print(f"  [Agente IA] ERROR: Gemini no devolvió JSON válido en extracción inicial. Detalles: {json_e}")
                print(f"  [Agente IA] Cadena JSON intentada: {extracted_json_str}")
                continue
            except Exception as parse_e:
                print(f"  [Agente IA] ERROR Inesperado al parsear JSON en extracción inicial: {parse_e}")
                print(f"  [Agente IA] Cadena JSON intentada: {extracted_json_str}")
                continue
        
        except ResourceExhausted as re:
            print(f"  [Agente IA] Error de Cuota/Recurso (ResourceExhausted) en extracción inicial: {re}")
            print(f"  [Agente IA] Posiblemente excediste los límites de tokens o de solicitudes del modelo.")
            print(f"  [Agente IA] Considera habilitar la facturación en Google Cloud o esperar a que las cuotas se restablezcan.")
            break # Rompemos el bucle principal si hay un error de cuota grave
        except Exception as e:
            print(f"  [Agente IA] Error general al interactuar con Gemini en extracción inicial: {e}")
            if 'response' in locals() and hasattr(response, 'text'):
                print(f"  [Agente IA] Respuesta cruda de Gemini (si hubo): {response.text}")
            continue

    if collected_candidates_data:
        final_message = f"Recolección de datos completada. Se encontraron {len(collected_candidates_data)} candidatos únicos."
        print(f"\n[Agente IA] TAREA COMPLETADA: {final_message}")
        for data in collected_candidates_data:
            print(data)
        return collected_candidates_data, final_message
    else:
        return [], "La recolección de datos no produjo resultados útiles."

# --- Rutas de Flask ---
@app.route('/')
def index():
    return render_template('index.html', candidates_data=[], message=None)

@app.route('/iniciar_recoleccion', methods=['POST'])
def iniciar_recoleccion():
    target_election = request.form.get('eleccion', 'elecciones presidenciales Colombia 2026')
    
    candidates_info, message = run_ai_agent_for_data_collection(target_election)
    
    return render_template('index.html', candidates_data=candidates_info, message=message)

if __name__ == '__main__':
    app.run(debug=True)