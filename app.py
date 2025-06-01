from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from main import LegalQASystem
import pandas as pd

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
        return jsonify({
            'status': 'success',
            'answer': answer,
            'feedback_id': str(pd.Timestamp.now())  # Egyedi azonosító a visszajelzéshez
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Felhasználói visszajelzés kezelése
    
    Returns:
        JSON válasz a visszajelzés feldolgozásáról
    """
    data = request.json
    if not data or 'feedback_id' not in data or 'rating' not in data:
        return jsonify({'status': 'error', 'message': 'Hiányzó visszajelzési adatok!'})
    
    try:
        qa_system.user_feedback.add_feedback(
            question=data.get('question', ''),
            answer=data.get('answer', ''),
            feedback={
                'rating': data['rating'],
                'comments': data.get('comments', ''),
                'feedback_id': data['feedback_id']
            }
        )
        return jsonify({'status': 'success', 'message': 'Visszajelzés sikeresen mentve!'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/feedback/analysis', methods=['GET'])
def feedback_analysis():
    """
    Visszajelzések elemzésének lekérdezése
    
    Returns:
        JSON válasz az elemzési eredményekkel
    """
    try:
        analysis = qa_system.get_feedback_analysis()
        return jsonify({
            'status': 'success',
            'analysis': analysis
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/feedback/analysis/temporal', methods=['GET'])
def temporal_analysis():
    """
    Időbeli trendek elemzése
    
    Returns:
        JSON válasz az időbeli trendekkel
    """
    try:
        days = request.args.get('days', default=30, type=int)
        analysis = qa_system.feedback_analyzer.analyze_temporal_trends(days)
        return jsonify({
            'status': 'success',
            'analysis': analysis
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/feedback/analysis/questions', methods=['GET'])
def question_analysis():
    """
    Kérdési mintázatok elemzése
    
    Returns:
        JSON válasz a kérdési mintázatokkal
    """
    try:
        analysis = qa_system.feedback_analyzer.analyze_question_patterns()
        return jsonify({
            'status': 'success',
            'analysis': analysis
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/feedback/analysis/quality', methods=['GET'])
def quality_analysis():
    """
    Válaszok minőségének elemzése
    
    Returns:
        JSON válasz a minőségi elemzéssel
    """
    try:
        analysis = qa_system.feedback_analyzer.analyze_answer_quality()
        return jsonify({
            'status': 'success',
            'analysis': analysis
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 