import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import json
from google.api_core.exceptions import ResourceExhausted

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your_secret_key')

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# --- Custom Search con más resultados ---
def google_custom_search(query, num_results=10):
    api_key = os.getenv("CUSTOM_SEARCH_API_KEY")
    cx_id = os.getenv("CUSTOM_SEARCH_ENGINE_ID")

    if not api_key or not cx_id:
        print("Error: CUSTOM_SEARCH_API_KEY o CUSTOM_SEARCH_ENGINE_ID no configurados")
        return []

    url = (
        f"https://www.googleapis.com/customsearch/v1?"
        f"key={api_key}&cx={cx_id}&q={query}&num={num_results}&cr=countryCO"
    )

    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        return [item['link'] for item in data.get('items', [])]
    except requests.exceptions.RequestException as e:
        print(f"Error en custom search: {e}")
        return []

def scrape_and_extract_text(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0'
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        for s in soup(["script", "style"]):
            s.extract()
        text = '\n'.join(
            phrase.strip()
            for line in soup.get_text().splitlines()
            for phrase in line.split("  ")
            if phrase.strip()
        )
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error scraper: {e}")
        return None

# --- Enriquecimiento por candidato ---
def enrich_candidate_data(candidate_info: dict, candidate_name: str, initial_url: str):
    fields = ['birth_date', 'education', 'experience', 'main_proposals']
    if not candidate_info.get('source_url'):
        candidate_info['source_url'] = initial_url
    for field in fields:
        is_missing = (
            candidate_info.get(field) in (None, 'null')
            or (field == 'main_proposals' and not candidate_info.get(field))
        )
        if not is_missing:
            continue

        prompt_field = {
            'birth_date': f"Fecha de nacimiento de {candidate_name} Colombia",
            'education': f"Estudios de {candidate_name} Colombia",
            'experience': f"Experiencia política de {candidate_name} Colombia",
            'main_proposals': f"Propuestas principales de {candidate_name} Colombia"
        }[field]

        for url in google_custom_search(prompt_field, num_results=5)[:2]:
            if url == initial_url:
                continue
            text = scrape_and_extract_text(url)
            if not text:
                continue
            embedding = {
                "role": "user",
                "parts": [{"text": f"""
Extrae en formato JSON el campo '{field}' de este texto sobre {candidate_name}:
---
{text[:3000]}
---
Devuélvelo solo como {{ "{field}": ... }}
"""}]
            }
            try:
                model = genai.GenerativeModel('models/gemini-2.0-flash-lite')
                resp = model.generate_content(contents=[embedding])
                j = resp.text.strip().lstrip('```json').rstrip('```')
                data = json.loads(j)
                val = data.get(field)
                if val and (field != 'main_proposals' or val != []):
                    candidate_info[field] = val
                    break
            except Exception as e:
                print(f"Error enriquecimiento {field}: {e}")
    return candidate_info

# --- Agente principal ---
def run_ai_agent_for_data_collection(target_election: str):
    search_query = f"principales candidatos {target_election} Colombia"
    urls = google_custom_search(search_query, num_results=10)

    collected = []
    names = set()
    urls_checked = 0
    max_urls = 30

    for url in urls:
        if len(collected) >= 5 or urls_checked >= max_urls:
            break
        urls_checked += 1

        text = scrape_and_extract_text(url)
        if not text:
            continue

        prompt = f"""
Del siguiente texto sobre candidatos políticos, extrae una lista JSON con hasta 5 objetos:
[{{full_name, political_party, birth_date, education, experience, main_proposals, source_url}}]
Solo retorna candidatos colombianos, nada más.
---
{text[:4000]}
---
"""
        try:
            model = genai.GenerativeModel('models/gemini-2.0-flash-lite')
            resp = model.generate_content(contents=[{"role":"user","parts":[{"text":prompt}]}])
            j = resp.text.strip().lstrip('```json').rstrip('```')
            candidates = json.loads(j)
            for c in candidates:
                name = c.get('full_name')
                party = c.get('political_party')
                if not name or name.lower() == 'null':
                    continue
                key = name.strip().lower()
                if key in names:
                    continue
                names.add(key)
                enriched = enrich_candidate_data(c, name, url)
                collected.append(enriched)
                if len(collected) >= 5:
                    break
        except Exception as e:
            print(f"Error extracción inicial: {e}")
            continue

    msg = f"Se encontraron {len(collected)} candidatos."
    return collected, msg

@app.route('/')
def index():
    return render_template('index.html', candidates_data=[], message=None)

@app.route('/iniciar_recoleccion', methods=['POST'])
def iniciar_recoleccion():
    elec = request.form.get('eleccion', 'elecciones presidenciales Colombia 2026')
    data, msg = run_ai_agent_for_data_collection(elec)
    return render_template('index.html', candidates_data=data, message=msg)

if __name__ == '__main__':
    app.run(debug=True)
