import sqlite3
from datetime import datetime
import json
from langchain_openai import ChatOpenAI


class UserFeedback:
    """Felhasználói visszajelzések kezelése"""

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self._init_database()

    def _init_database(self):
        """Adatbázis inicializálása"""
        conn = sqlite3.connect('feedback.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                feedback_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comments TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def add_feedback(self, question: str, answer: str, feedback: dict):
        """
        Új visszajelzés hozzáadása az adatbázisba
        """
        conn = sqlite3.connect('feedback.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO feedback (feedback_id, question, answer, rating, comments)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            feedback['feedback_id'],
            question,
            answer,
            feedback['rating'],
            feedback.get('comments', '')
        ))
        conn.commit()
        conn.close()

    def analyze_feedback(self) -> dict:
        """
        Visszajelzések elemzése az adatbázisból
        """
        conn = sqlite3.connect('feedback.db')
        c = conn.cursor()

        # Alapvető statisztikák
        c.execute('SELECT AVG(rating) FROM feedback')
        avg_rating = c.fetchone()[0] or 0

        c.execute('SELECT rating, COUNT(*) FROM feedback GROUP BY rating')
        rating_distribution = dict(c.fetchall())

        c.execute('SELECT comments FROM feedback WHERE comments != ""')
        comments = [row[0] for row in c.fetchall()]

        conn.close()

        analysis_prompt = f"""
        Elemzd a következő visszajelzési adatokat:
        
        Átlagos értékelés: {avg_rating}
        Értékelési eloszlás: {rating_distribution}
        Megjegyzések: {comments}
        
        Add meg az elemzési eredményt JSON formátumban:
        {{
            "atlagos_ertekelés": float,
            "ertekelési_eloszlás": dict,
            "gyakori_hibak": ["hiba1", "hiba2", ...],
            "javaslatok": ["javaslat1", "javaslat2", ...],
            "trendek": ["trend1", "trend2", ...],
            "minőségi_javulás": bool,
            "javulási_területek": ["terület1", "terület2", ...]
        }}
        """

        try:
            analysis_result = self.llm.invoke(analysis_prompt)
            return json.loads(analysis_result.content)
        except (json.JSONDecodeError, TypeError, ValueError):
            print(f"Hiba a visszajelzések JSON elemzése során.")
            return {}
        except Exception as e:
            print(f"Hiba a visszajelzések elemzése során: {str(e)}")
            return {}


class FeedbackAnalyzer:
    """Részletes visszajelzés-elemző"""

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)

    def analyze_temporal_trends(self, days: int = 30) -> dict:
        """
        Időbeli trendek elemzése
        """
        conn = sqlite3.connect('feedback.db')
        c = conn.cursor()

        c.execute('''
            SELECT 
                date(timestamp) as date,
                AVG(rating) as avg_rating,
                COUNT(*) as feedback_count
            FROM feedback
            WHERE timestamp >= datetime('now', ?)
            GROUP BY date(timestamp)
            ORDER BY date
        ''', (f'-{days} days',))

        temporal_data = c.fetchall()
        conn.close()

        analysis_prompt = f"""
        Elemzd a következő időbeli trendeket: {temporal_data}
        Add meg az elemzési eredményt JSON formátumban.
        """

        try:
            analysis_result = self.llm.invoke(analysis_prompt)
            return json.loads(analysis_result.content)
        except (json.JSONDecodeError, TypeError, ValueError):
            print(f"Hiba az időbeli trendek JSON elemzése során.")
            return {}
        except Exception as e:
            print(f"Hiba az időbeli trendek elemzése során: {str(e)}")
            return {}

    def analyze_question_patterns(self) -> dict:
        """
        Kérdési mintázatok elemzése
        """
        conn = sqlite3.connect('feedback.db')
        c = conn.cursor()

        c.execute('SELECT question, rating FROM feedback')
        questions_data = c.fetchall()
        conn.close()

        analysis_prompt = f"""
        Elemzd a következő kérdési mintázatokat: {questions_data}
        Add meg az elemzési eredményt JSON formátumban.
        """

        try:
            analysis_result = self.llm.invoke(analysis_prompt)
            return json.loads(analysis_result.content)
        except (json.JSONDecodeError, TypeError, ValueError):
            print(f"Hiba a kérdési mintázatok JSON elemzése során.")
            return {}
        except Exception as e:
            print(f"Hiba a kérdési mintázatok elemzése során: {str(e)}")
            return {}

    def analyze_answer_quality(self) -> dict:
        """
        Válaszok minőségének elemzése
        """
        conn = sqlite3.connect('feedback.db')
        c = conn.cursor()

        c.execute('SELECT answer, rating, comments FROM feedback')
        answers_data = c.fetchall()
        conn.close()

        analysis_prompt = f"""
        Elemzd a következő válaszok minőségét: {answers_data}
        Add meg az elemzési eredményt JSON formátumban.
        """

        try:
            analysis_result = self.llm.invoke(analysis_prompt)
            return json.loads(analysis_result.content)
        except (json.JSONDecodeError, TypeError, ValueError):
            print(f"Hiba a válaszok minőségének JSON elemzése során.")
            return {}
        except Exception as e:
            print(f"Hiba a válaszok minőségének elemzése során: {str(e)}")
            return {} 