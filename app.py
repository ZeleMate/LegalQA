from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from main import LegalQASystem
import os

app = Flask(__name__)
CORS(app)  # CORS engedélyezése minden route-hoz
app.config['SECRET_KEY'] = os.urandom(24)

# QA rendszer inicializálása
qa_system = LegalQASystem()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/load_documents', methods=['POST'])
def load_documents():
    try:
        success = qa_system.load_documents()
        if success:
            return jsonify({'status': 'success', 'message': 'Dokumentumok sikeresen betöltve!'})
        else:
            return jsonify({'status': 'error', 'message': 'Hiba történt a dokumentumok betöltése során!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/ask', methods=['POST'])
def ask():
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