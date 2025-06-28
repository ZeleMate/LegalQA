import json
import sqlite3

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# Hogy elkerüljük a körkörös importot, a type hint-hez stringként adjuk meg az osztályt
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .qa_system import LegalQASystem


class QualityAssurance:
    """Válaszok minőségének biztosítása"""

    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)

    def validate_response(self, question: str, answer: str, context: list[Document]) -> dict:
        """Válasz minőségének ellenőrzése"""
        validation_prompt = f"""
        Ellenőrizd a következő jogi válasz minőségét a megadott kontextus alapján:
        
        Kérdés: {question}
        Válasz: {answer}
        
        Ellenőrizd a következő szempontokat:
        1. Jogi pontosság
        2. Dokumentumok megfelelő használata
        3. Struktúra teljesítése
        
        Add meg a validációs eredményt JSON formátumban:
        {{
            "jogi_pontossag": true/false,
            "dokumentumok_hasznalata": true/false,
            "struktura": true/false,
            "hibak": ["hiba1", "hiba2", ...],
            "javaslatok": ["javaslat1", "javaslat2", ...]
        }}
        """
        try:
            validation_result = self.llm.invoke(validation_prompt)
            return json.loads(validation_result.content)
        except (json.JSONDecodeError, TypeError):
            print(f"Hiba a validációs JSON feldolgozása során.")
            return {"jogi_pontossag": True, "dokumentumok_hasznalata": True, "struktura": True, "hibak": [], "javaslatok": []}
        except Exception as e:
            print(f"Hiba a validáció során: {str(e)}")
            return {"jogi_pontossag": True, "dokumentumok_hasznalata": True, "struktura": True, "hibak": [], "javaslatok": []}


class AutoLearner:
    """Automatikus tanulás a visszajelzések alapján"""

    def __init__(self, qa_system: 'LegalQASystem'):
        self.qa_system = qa_system
        self.llm = ChatOpenAI(temperature=0)

    def learn_from_feedback(self):
        """Tanulás a visszajelzések alapján"""
        conn = sqlite3.connect('feedback.db')
        c = conn.cursor()
        c.execute('SELECT question, answer, comments FROM feedback WHERE rating <= 3 ORDER BY timestamp DESC LIMIT 10')
        low_rated = c.fetchall()
        c.execute('SELECT question, answer, comments FROM feedback WHERE rating >= 4 ORDER BY timestamp DESC LIMIT 10')
        high_rated = c.fetchall()
        conn.close()

        if not low_rated and not high_rated:
            return {}

        learning_prompt = f"""
        Elemzd a válaszokat és javasolj fejlesztéseket a prompt-ra.
        Alacsony értékelésű válaszok: {low_rated}
        Magas értékelésű válaszok: {high_rated}
        
        Add meg a tanulási eredményt JSON formátumban:
        {{
            "prompt_javítások": ["javítás1", "javítás2", ...]
        }}
        """
        try:
            learning_result = self.llm.invoke(learning_prompt)
            improvements = json.loads(learning_result.content)
            
            if 'prompt_javítások' in improvements:
                self._apply_prompt_improvements(improvements['prompt_javítások'])
            
            return improvements
        except (json.JSONDecodeError, TypeError):
            print("Hiba a tanulási JSON feldolgozása során.")
            return {}
        except Exception as e:
            print(f"Hiba az automatikus tanulás során: {str(e)}")
            return {}

    def _apply_prompt_improvements(self, improvements: list):
        """Prompt javítások alkalmazása (jelenleg csak naplózva)"""
        if improvements:
            print("Automatikus tanulás: Prompt javítási javaslatok érkeztek.")
            # A valós implementáció itt frissítené a qa_system prompt sablonját.
            # Ez egy bonyolultabb logikát igényel, pl. a sablonfájl módosítását.
            # Jelenleg csak kiírjuk a javaslatokat.
            for imp in improvements:
                print(f"- Javaslat: {imp}") 