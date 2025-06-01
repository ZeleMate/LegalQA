from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from main import LegalQASystem

app = Flask(__name__)
CORS(app)  # CORS engedélyezése minden route-hoz

# Jogi kérdés-válasz rendszer inicializálása
qa_system = LegalQASystem()

@app.route('/')
def home():
    """Főoldal megjelenítése"""
    return render_template('index.html')

@app.route('/load_documents', methods=['POST'])
def load_documents():
    """
    Dokumentumok betöltése a rendszerbe
    
    Returns:
        JSON válasz a betöltés sikerességéről
    """
    try:
        success = qa_system.load_documents()
        return jsonify({
            'status': 'success' if success else 'error',
            'message': 'Dokumentumok sikeresen betöltve!' if success else 'Hiba történt a dokumentumok betöltése során!'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/ask', methods=['POST'])
def ask():
    """
    Kérdés megválaszolása a rendszer által
    
    Returns:
        JSON válasz a kérdésre
    """
    question = request.json.get('question')
    if not question:
        return jsonify({'status': 'error', 'message': 'Kérdés megadása kötelező!'})
    
    try:
        answer = qa_system.ask_question(question)
        return jsonify({'status': 'success', 'answer': answer})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 