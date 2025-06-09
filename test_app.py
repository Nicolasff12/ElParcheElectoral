import pytest
import json
from unittest.mock import patch, MagicMock
from app import app, google_custom_search, scrape_and_extract_text, run_ai_agent_for_data_collection, enrich_candidate_data

# Configuración de pytest para la aplicación Flask
@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# --- Tests para google_custom_search ---
def test_google_custom_search_success():
    # Simular una respuesta exitosa de la API de Google Custom Search
    mock_search_results = {
        'items': [
            {'link': 'http://example.com/url1'},
            {'link': 'http://example.com/url2'}
        ]
    }
    with patch('requests.get') as mock_get:
        mock_get.return_value.raise_for_status.return_value = None # Simula respuesta 200 OK
        mock_get.return_value.json.return_value = mock_search_results
        
        # Simular las variables de entorno para la prueba
        with patch.dict(os.environ, {'CUSTOM_SEARCH_API_KEY': 'dummy_key', 'CUSTOM_SEARCH_ENGINE_ID': 'dummy_cx'}):
            urls = google_custom_search("test query", num_results=2)
            assert urls == ['http://example.com/url1', 'http://example.com/url2']
            mock_get.assert_called_once() # Asegurarse de que requests.get fue llamado

def test_google_custom_search_api_error():
    # Simular un error HTTP de la API de Google Custom Search (ej. 400 Bad Request)
    with patch('requests.get') as mock_get:
        mock_get.return_value.raise_for_status.side_effect = requests.exceptions.RequestException("400 Client Error")
        
        with patch.dict(os.environ, {'CUSTOM_SEARCH_API_KEY': 'dummy_key', 'CUSTOM_SEARCH_ENGINE_ID': 'dummy_cx'}):
            urls = google_custom_search("test query")
            assert urls == [] # Debería devolver una lista vacía en caso de error
            mock_get.assert_called_once()

def test_google_custom_search_no_env_vars():
    # Simular que las variables de entorno no están configuradas
    with patch.dict(os.environ, {}, clear=True): # Limpiar todas las variables de entorno para esta prueba
        urls = google_custom_search("test query")
        assert urls == [] # Debería devolver lista vacía si las variables no están
        # No deberíamos llamar a requests.get si las variables no están
        with patch('requests.get') as mock_get:
            google_custom_search("test query")
            mock_get.assert_not_called()


# --- Tests para scrape_and_extract_text ---
def test_scrape_and_extract_text_success():
    # Simular una página HTML sencilla
    mock_html = """
    <html>
    <head><title>Test Page</title></head>
    <body>
        <p>This is some text.</p>
        <script>var x = 1;</script>
        <style>body { color: red; }</style>
        <div>More text here.</div>
        <p>    Whitespace      test.   </p>
    </body>
    </html>
    """
    with patch('requests.get') as mock_get:
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.text = mock_html
        
        extracted_text = scrape_and_extract_text("http://example.com/test_page")
        expected_text = "Test Page\nThis is some text.\nMore text here.\nWhitespace test."
        assert extracted_text == expected_text
        mock_get.assert_called_once_with("http://example.com/test_page", headers=ANY, timeout=10) # ANY para no preocuparnos por el User-Agent exacto

def test_scrape_and_extract_text_http_error():
    # Simular un error HTTP durante el scraping
    with patch('requests.get') as mock_get:
        mock_get.return_value.raise_for_status.side_effect = requests.exceptions.RequestException("404 Not Found")
        
        extracted_text = scrape_and_extract_text("http://example.com/non_existent_page")
        assert extracted_text is None
        mock_get.assert_called_once()

# --- Tests para run_ai_agent_for_data_collection (ejemplo básico) ---
def test_run_ai_agent_success(mocker):
    # Mockear google_custom_search para devolver URLs de prueba
    mocker.patch('app.google_custom_search', return_value=['http://example.com/candidate1', 'http://example.com/candidate2'])
    
    # Mockear scrape_and_extract_text para devolver contenido simulado
    mock_page_text_c1 = "Text about Juan Perez. Political party: Liberal. Proposals: build bridges. Experience: congressman."
    mock_page_text_c2 = "Text about Maria Lopez. Political party: Conservative. Proposals: improve education. Experience: senator."
    mocker.patch('app.scrape_and_extract_text', side_effect=[mock_page_text_c1, mock_page_text_c2])

    # Mockear genai.GenerativeModel para simular la respuesta de Gemini
    # Usamos MagicMock para simular la cadena de llamadas .generate_content().text
    mock_gemini_response_c1 = MagicMock()
    mock_gemini_response_c1.text = json.dumps({
        "full_name": "Juan Perez",
        "political_party": "Liberal",
        "birth_date": "null",
        "education": "null",
        "experience": "congressman",
        "main_proposals": ["build bridges"],
        "cv_link": "null",
        "alliances": []
    })

    mock_gemini_response_c2 = MagicMock()
    mock_gemini_response_c2.text = json.dumps({
        "full_name": "Maria Lopez",
        "political_party": "Conservative",
        "birth_date": "null",
        "education": "null",
        "experience": "senator",
        "main_proposals": ["improve education"],
        "cv_link": "null",
        "alliances": []
    })

    # Simular las respuestas de Gemini para la extracción inicial
    mocker.patch('google.generativeai.GenerativeModel.generate_content', side_effect=[mock_gemini_response_c1, mock_gemini_response_c2, mock_gemini_response_c1, mock_gemini_response_c2])
    # ^ Se llama 2 veces para la extracción inicial, y 2 veces más para el enriquecimiento (uno por candidato para cada campo que falta). Ajustar si cambias campos.

    # También necesitamos simular las variables de entorno para Gemini
    with patch.dict(os.environ, {'GOOGLE_API_KEY': 'dummy_gemini_key'}):
        candidates, message = run_ai_agent_for_data_collection("elecciones presidenciales Colombia 2026")
        
        assert len(candidates) == 2
        assert candidates[0]['full_name'] == "Juan Perez"
        assert candidates[1]['full_name'] == "Maria Lopez"
        assert "completada" in message.lower()
        
        # Verificar que scrape_and_extract_text fue llamado por cada URL
        assert app.scrape_and_extract_text.call_count == 2 # 2 URLs iniciales

        # Verificar que Gemini fue llamado
        assert google.generativeai.GenerativeModel.generate_content.call_count >= 2 # Al menos 2 llamadas iniciales

# --- Tests para enrich_candidate_data ---
def test_enrich_candidate_data_adds_missing_info(mocker):
    # Mock para la búsqueda de enriquecimiento
    mocker.patch('app.google_custom_search', return_value=['http://example.com/enrich_url'])
    
    # Mock para el scraping de la URL de enriquecimiento
    mock_enrich_text = "Juan Perez studied at Uni. Alliances: Green Alliance."
    mocker.patch('app.scrape_and_extract_text', return_value=mock_enrich_text)

    # Mock para la respuesta de Gemini durante el enriquecimiento
    mock_gemini_enrich_response_edu = MagicMock()
    mock_gemini_enrich_response_edu.text = json.dumps({"education": "University Degree"})
    
    mock_gemini_enrich_response_alliance = MagicMock()
    mock_gemini_enrich_response_alliance.text = json.dumps({"alliances": ["Green Alliance"]})

    # Simular que Gemini es llamado para cada campo faltante
    mocker.patch('google.generativeai.GenerativeModel.generate_content', side_effect=[
        mock_gemini_enrich_response_edu, # Para 'education'
        MagicMock(text=json.dumps({"experience": "Ex-President"})), # Para 'experience'
        MagicMock(text=json.dumps({"cv_link": "http://cv.com/juan"})), # Para 'cv_link'
        mock_gemini_enrich_response_alliance # Para 'alliances'
    ])
    
    # Candidato inicial con datos faltantes
    initial_candidate_info = {
        "full_name": "Juan Perez",
        "political_party": "Partido X",
        "birth_date": None, # Faltante
        "education": "null", # Faltante
        "experience": None, # Faltante
        "main_proposals": [],
        "cv_link": None, # Faltante
        "alliances": [] # Faltante
    }

    # Simular la variable de entorno de Gemini
    with patch.dict(os.environ, {'GOOGLE_API_KEY': 'dummy_gemini_key'}):
        enriched_data = enrich_candidate_data(initial_candidate_info.copy(), "Juan Perez", "http://original.com")
        
        assert enriched_data['education'] == "University Degree"
        assert enriched_data['experience'] == "Ex-President"
        assert enriched_data['cv_link'] == "http://cv.com/juan"
        assert enriched_data['alliances'] == ["Green Alliance"]
        assert enriched_data['birth_date'] is None # No se buscó, debería seguir siendo None/null si no se añadió al prompt
        assert enriched_data['source_url'] == "http://original.com" # Debe mantener la original

        # Verificar que se hicieron llamadas para los campos faltantes
        assert app.google_custom_search.call_count == 4 # Una por cada campo 'education', 'experience', 'cv_link', 'alliances'
        assert app.scrape_and_extract_text.call_count == 4 # Una por cada campo
        assert google.generativeai.GenerativeModel.generate_content.call_count == 4 # Una por cada campo

# --- Test para el endpoint Flask ---
from urllib.parse import unquote_plus
from flask import url_for

def test_index_page(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Elecciones Presidenciales" in response.data

@patch('app.run_ai_agent_for_data_collection')
def test_iniciar_recoleccion_post(mock_run_agent, client):
    # Simular que el agente devuelve algunos datos
    mock_run_agent.return_value = ([{'full_name': 'Test Candidate', 'political_party': 'Test Party'}], "Recolección exitosa")
    
    response = client.post('/iniciar_recoleccion', data={'eleccion': 'Elecciones Ficticias'})
    assert response.status_code == 200
    assert b"Recolecci\xc3\xb3n exitosa" in response.data # html entity for 'ó'
    assert b"Test Candidate" in response.data
    mock_run_agent.assert_called_once_with('Elecciones Ficticias')

# Para la importación de ANY en scrape_and_extract_text
from unittest.mock import ANY