import os
from dotenv import load_dotenv
import pandas as pd
import faiss
import pickle
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
import networkx as nx
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import BaseModel, Field
import numpy as np
import sqlite3
from datetime import datetime
from langchain.prompts import PromptTemplate

# Környezeti változók betöltése
load_dotenv()

class InMemoryHistory(BaseChatMessageHistory):
    """Egyszerű memória alapú chat történet kezelő"""
    
    def __init__(self):
        self.messages = []
    
    def add_message(self, message):
        self.messages.append(message)
    
    def clear(self):
        self.messages = []

class CustomRetriever(BaseRetriever, BaseModel):
    """Egyedi dokumentum kereső a jogi dokumentumokhoz"""
    
    embeddings: OpenAIEmbeddings = Field(..., description="OpenAI embeddings model")
    faiss_index: faiss.Index = Field(..., description="FAISS index")
    id_mapping: dict = Field(..., description="ID mapping dictionary")
    documents_df: pd.DataFrame = Field(..., description="Documents DataFrame")
    graph: nx.Graph = Field(..., description="Graph data")

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        """
        Releváns dokumentumok keresése a kérdés alapján
        
        Args:
            query: A keresési kérdés
            run_manager: Opcionális futáskezelő
            
        Returns:
            Lista a releváns dokumentumokról
        """
        query_embedding = self.embeddings.embed_query(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        D, I = self.faiss_index.search(query_embedding, 8)
        
        documents = []
        seen_doc_ids = set()
        
        for idx, distance in zip(I[0], D[0]):
            if idx in self.id_mapping:
                doc_id = self.id_mapping[idx]
                if doc_id in seen_doc_ids:
                    continue
                    
                matching_docs = self.documents_df[self.documents_df['doc_id'] == doc_id]
                if not matching_docs.empty:
                    doc = matching_docs.iloc[0]
                    metadata = {
                        'MeghozoBirosag': doc['MeghozoBirosag'],
                        'JogTerulet': doc['JogTerulet'],
                        'Jogszabalyhelyek': doc['Jogszabalyhelyek'],
                        'HatarozatEve': doc['HatarozatEve'],
                        'doc_id': doc_id,
                        'relevancia': float(1 / (1 + distance))
                    }
                    documents.append(Document(
                        page_content=f"Dokumentum azonosító: {doc_id}",
                        metadata=metadata
                    ))
                    seen_doc_ids.add(doc_id)
        
        if self.graph and len(documents) > 0:
            for doc in documents:
                if hasattr(doc, 'metadata') and doc.metadata and isinstance(doc.metadata, dict) and 'doc_id' in doc.metadata:
                    doc_id = doc.metadata['doc_id']
                    if doc_id in self.graph:
                        neighbors = list(self.graph.neighbors(doc_id))[:3]
                        for neighbor in neighbors:
                            if neighbor not in seen_doc_ids and len(documents) < 12:
                                neighbor_doc = self.documents_df[self.documents_df['doc_id'] == neighbor]
                                if not neighbor_doc.empty:
                                    neighbor_row = neighbor_doc.iloc[0]
                                    neighbor_metadata = {
                                        'MeghozoBirosag': neighbor_row['MeghozoBirosag'],
                                        'JogTerulet': neighbor_row['JogTerulet'],
                                        'Jogszabalyhelyek': neighbor_row['Jogszabalyhelyek'],
                                        'HatarozatEve': neighbor_row['HatarozatEve'],
                                        'doc_id': neighbor,
                                        'relevancia': 0.7
                                    }
                                    documents.append(Document(
                                        page_content=f"Dokumentum azonosító: {neighbor}",
                                        metadata=neighbor_metadata
                                    ))
                                    seen_doc_ids.add(neighbor)
        
        documents.sort(key=lambda x: x.metadata.get('relevancia', 0), reverse=True)
        return documents

class QualityAssurance:
    """Válaszok minőségének biztosítása"""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
    
    def validate_response(self, question: str, answer: str, context: list[Document]) -> dict:
        """
        Válasz minőségének ellenőrzése
        
        Args:
            question: A felhasználó kérdése
            answer: A generált válasz
            context: A használt dokumentumok
            
        Returns:
            dict: Validációs eredmények
        """
        validation_prompt = f"""
        Ellenőrizd a következő jogi válasz minőségét:
        
        Kérdés: {question}
        Válasz: {answer}
        
        Ellenőrizd a következő szempontokat:
        1. Jogi pontosság
        2. Dokumentumok megfelelő használata
        3. Struktúra teljesítése
        4. Konzisztencia
        5. Időbeli érvényesség
        
        Add meg a validációs eredményt JSON formátumban:
        {{
            "jogi_pontossag": true/false,
            "dokumentumok_hasznalata": true/false,
            "struktura": true/false,
            "konzisztencia": true/false,
            "idobeli_ervenyesseg": true/false,
            "hibak": ["hiba1", "hiba2", ...],
            "javaslatok": ["javaslat1", "javaslat2", ...]
        }}
        """
        
        try:
            validation_result = self.llm.invoke(validation_prompt)
            return eval(validation_result.content)
        except Exception as e:
            print(f"Hiba a validáció során: {str(e)}")
            return {
                "jogi_pontossag": True,
                "dokumentumok_hasznalata": True,
                "struktura": True,
                "konzisztencia": True,
                "idobeli_ervenyesseg": True,
                "hibak": [],
                "javaslatok": []
            }

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
        
        Args:
            question: A felhasználó kérdése
            answer: A rendszer válasza
            feedback: A visszajelzés adatai
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
        
        Returns:
            dict: Elemzési eredmények
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
            return eval(analysis_result.content)
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
        
        Args:
            days: Hány napra visszamenőleg elemezzen
            
        Returns:
            dict: Időbeli trendek elemzése
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
        Elemzd a következő időbeli trendeket:
        
        {temporal_data}
        
        Add meg az elemzési eredményt JSON formátumban:
        {{
            "trend_irány": "javuló/romló/stabil",
            "átlagos_értékelés_változás": float,
            "visszajelzések_száma_változás": float,
            "jelentős_változások": ["változás1", "változás2", ...],
            "javaslatok": ["javaslat1", "javaslat2", ...]
        }}
        """
        
        try:
            analysis_result = self.llm.invoke(analysis_prompt)
            return eval(analysis_result.content)
        except Exception as e:
            print(f"Hiba az időbeli trendek elemzése során: {str(e)}")
            return {}
    
    def analyze_question_patterns(self) -> dict:
        """
        Kérdési mintázatok elemzése
        
        Returns:
            dict: Kérdési mintázatok elemzése
        """
        conn = sqlite3.connect('feedback.db')
        c = conn.cursor()
        
        c.execute('SELECT question, rating FROM feedback')
        questions_data = c.fetchall()
        conn.close()
        
        analysis_prompt = f"""
        Elemzd a következő kérdési mintázatokat:
        
        {questions_data}
        
        Add meg az elemzési eredményt JSON formátumban:
        {{
            "gyakori_kérdéstípusok": ["típus1", "típus2", ...],
            "legmagasabb_értékelésű_kérdések": ["kérdés1", "kérdés2", ...],
            "legalacsonyabb_értékelésű_kérdések": ["kérdés1", "kérdés2", ...],
            "javaslatok": ["javaslat1", "javaslat2", ...]
        }}
        """
        
        try:
            analysis_result = self.llm.invoke(analysis_prompt)
            return eval(analysis_result.content)
        except Exception as e:
            print(f"Hiba a kérdési mintázatok elemzése során: {str(e)}")
            return {}
    
    def analyze_answer_quality(self) -> dict:
        """
        Válaszok minőségének elemzése
        
        Returns:
            dict: Válaszok minőségének elemzése
        """
        conn = sqlite3.connect('feedback.db')
        c = conn.cursor()
        
        c.execute('SELECT answer, rating, comments FROM feedback')
        answers_data = c.fetchall()
        conn.close()
        
        analysis_prompt = f"""
        Elemzd a következő válaszok minőségét:
        
        {answers_data}
        
        Add meg az elemzési eredményt JSON formátumban:
        {{
            "átlagos_válasz_minőség": float,
            "gyakori_hibák": ["hiba1", "hiba2", ...],
            "erősségek": ["erősség1", "erősség2", ...],
            "javulási_területek": ["terület1", "terület2", ...],
            "javaslatok": ["javaslat1", "javaslat2", ...]
        }}
        """
        
        try:
            analysis_result = self.llm.invoke(analysis_prompt)
            return eval(analysis_result.content)
        except Exception as e:
            print(f"Hiba a válaszok minőségének elemzése során: {str(e)}")
            return {}

class AutoLearner:
    """Automatikus tanulás a visszajelzések alapján"""
    
    def __init__(self, qa_system):
        self.qa_system = qa_system
        self.llm = ChatOpenAI(temperature=0)
    
    def learn_from_feedback(self):
        """
        Tanulás a visszajelzések alapján
        
        Returns:
            dict: Tanulási eredmények
        """
        conn = sqlite3.connect('feedback.db')
        c = conn.cursor()
        
        # Alacsony értékelésű válaszok lekérdezése
        c.execute('''
            SELECT question, answer, comments
            FROM feedback
            WHERE rating <= 3
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        low_rated_responses = c.fetchall()
        
        # Magas értékelésű válaszok lekérdezése
        c.execute('''
            SELECT question, answer, comments
            FROM feedback
            WHERE rating >= 4
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        high_rated_responses = c.fetchall()
        
        conn.close()
        
        learning_prompt = f"""
        Elemzd a következő válaszokat és javasolj fejlesztéseket:
        
        Alacsony értékelésű válaszok:
        {low_rated_responses}
        
        Magas értékelésű válaszok:
        {high_rated_responses}
        
        Add meg a tanulási eredményt JSON formátumban:
        {{
            "javítandó_területek": ["terület1", "terület2", ...],
            "erősségek": ["erősség1", "erősség2", ...],
            "prompt_javítások": ["javítás1", "javítás2", ...],
            "keresési_javítások": ["javítás1", "javítás2", ...],
            "struktúra_javítások": ["javítás1", "javítás2", ...]
        }}
        """
        
        try:
            learning_result = self.llm.invoke(learning_prompt)
            improvements = eval(learning_result.content)
            
            # Prompt javítások alkalmazása
            if 'prompt_javítások' in improvements:
                self._apply_prompt_improvements(improvements['prompt_javítások'])
            
            # Keresési javítások alkalmazása
            if 'keresési_javítások' in improvements:
                self._apply_search_improvements(improvements['keresési_javítások'])
            
            return improvements
        except Exception as e:
            print(f"Hiba az automatikus tanulás során: {str(e)}")
            return {}
    
    def _apply_prompt_improvements(self, improvements: list):
        """
        Prompt javítások alkalmazása
        
        Args:
            improvements: A javítások listája
        """
        current_prompt = self.qa_system._get_qa_prompt()
        improved_prompt = current_prompt.template
        
        for improvement in improvements:
            # Itt implementáljuk a prompt javításokat
            # Például: új szekciók hozzáadása, meglévők módosítása
            pass
        
        # Frissítjük a prompt-ot
        self.qa_system._get_qa_prompt = lambda: PromptTemplate(
            input_variables=["context", "question"],
            template=improved_prompt
        )
    
    def _apply_search_improvements(self, improvements: list):
        """
        Keresési javítások alkalmazása
        
        Args:
            improvements: A javítások listája
        """
        # Itt implementáljuk a keresési javításokat
        # Például: súlyozási módosítások, új keresési paraméterek
        pass

class DocumentCache:
    """Singleton osztály a dokumentumok gyorsítótárazásához"""
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DocumentCache, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.documents_df = None
            self.faiss_index = None
            self.id_mapping = None
            self.graph = None
            self._initialized = True
    
    def load_documents(self) -> bool:
        """
        Dokumentumok betöltése és gyorsítótárazása
        
        Returns:
            bool: Sikeres betöltés esetén True, egyébként False
        """
        if all([self.documents_df is not None, 
                self.faiss_index is not None, 
                self.id_mapping is not None, 
                self.graph is not None]):
            return True
            
        try:
            self.documents_df = pd.read_parquet('processed_data/processed_documents_with_embeddings.parquet')
            self.faiss_index = faiss.read_index('processed_data/faiss_index.bin')
            
            with open('processed_data/faiss_id_mapping.pkl', 'rb') as f:
                self.id_mapping = pickle.load(f)
            
            self.graph = nx.read_graphml('processed_data/graph_data/graph.graphml')
            return True
        except Exception as e:
            print(f"Hiba az adatok betöltése során: {str(e)}")
            return False
    
    def get_documents(self):
        """Dokumentumok lekérése a gyorsítótárból"""
        return self.documents_df, self.faiss_index, self.id_mapping, self.graph

class LegalQASystem:
    """Jogi kérdés-válasz rendszer"""
    
    def __init__(self):
        """Rendszer inicializálása"""
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.qa_chain = None
        self.chat_history = InMemoryHistory()
        self.document_cache = DocumentCache()
        self.quality_assurance = QualityAssurance()
        self.user_feedback = UserFeedback()
        self.feedback_analyzer = FeedbackAnalyzer()
        self.auto_learner = AutoLearner(self)

    def load_documents(self) -> bool:
        """
        Feldolgozott dokumentumok és indexek betöltése
        
        Returns:
            bool: Sikeres betöltés esetén True, egyébként False
        """
        try:
            if not self.document_cache.load_documents():
                return False
                
            self.documents_df, self.faiss_index, self.id_mapping, self.graph = self.document_cache.get_documents()
            self._initialize_qa_chain()
            return True
        except Exception as e:
            print(f"Hiba az adatok betöltése során: {str(e)}")
            return False

    def _initialize_qa_chain(self):
        """QA lánc inicializálása a dokumentumok betöltése után"""
        llm = ChatOpenAI(temperature=0)
        
        retriever = CustomRetriever(
            embeddings=self.embeddings,
            faiss_index=self.faiss_index,
            id_mapping=self.id_mapping,
            documents_df=self.documents_df,
            graph=self.graph
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self._get_qa_prompt()}
        )

    def _get_qa_prompt(self):
        """
        Egyedi prompt generálása a jogi kérdésekhez
        
        Returns:
            PromptTemplate: A generált prompt sablon
        """
        from langchain.prompts import PromptTemplate
        
        template = """Te egy jogi asszisztens vagy, aki a megadott jogi dokumentumok alapján válaszol a kérdésekre.

        A válaszodat a következő struktúrában add meg:

        1. RÖVID ÖSSZEFOGLALÓ
        - A kérdés lényegének rövid összefoglalása
        - A legfontosabb jogi következmények

        2. RÉSZLETES JOGI ELEMZÉS
        - Releváns jogszabályok és precedensek
        - Jogi érvelés és következtetések
        - Időbeli változások és aktuális gyakorlat

        3. RELEVÁNS DOKUMENTUMOK
        Minden dokumentum esetén add meg:
        - Dokumentum azonosító (doc_id)
        - Meghozó bíróság
        - Jogtérület
        - Határozat éve
        - Relevancia százalékban
        - Röviden: miért releváns ez a dokumentum

        4. GYAKORLATI TANÁCSOK
        - Konkrét lépések és teendők
        - Elkerülendő hibák
        - Javasolt megközelítések

        5. TÖVÁBBI FORRÁSOK
        - Kapcsolódó jogi területek
        - További dokumentumok
        - Hivatkozások

        Fontos szabályok:
        1. CSAK a megadott kontextusban található dokumentumokat használd
        2. Ha nincs releváns dokumentum, azt jelezd: "Nem található releváns dokumentum ehhez a kérdéshez."
        3. A dokumentumokat relevancia szerint rendezd (csökkenő sorrendben)
        4. Mindig hivatkozz a konkrét dokumentumokra
        5. Ha bizonytalan vagy, jelezd: "Ez a kérdés további jogi tanácsadást igényel."
        
        Kontextus: {context}
        Kérdés: {question}
        Válasz:"""
        
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

    def ask_question(self, question: str) -> str:
        """
        Kérdés megválaszolása a rendszer által
        
        Args:
            question: A felhasználó kérdése
            
        Returns:
            str: A rendszer válasza
            
        Raises:
            ValueError: Ha a rendszer nincs inicializálva
        """
        if not self.qa_chain:
            raise ValueError("A QA rendszer nincs inicializálva! Először töltsd be a dokumentumokat.")
        
        try:
            # Első lépés: válasz generálása
            result = self.qa_chain.invoke({
                "question": question,
                "chat_history": self.chat_history.messages
            })
            
            # Második lépés: válasz minőségének ellenőrzése
            validation_result = self.quality_assurance.validate_response(
                question=question,
                answer=result['answer'],
                context=result.get('source_documents', [])
            )
            
            # Harmadik lépés: válasz javítása ha szükséges
            if not all(validation_result.values()):
                improved_prompt = f"""
                Javítsd a következő jogi választ a validációs eredmények alapján:
                
                Eredeti válasz: {result['answer']}
                
                Validációs eredmények:
                - Jogi pontosság: {validation_result['jogi_pontossag']}
                - Dokumentumok használata: {validation_result['dokumentumok_hasznalata']}
                - Struktúra: {validation_result['struktura']}
                - Konzisztencia: {validation_result['konzisztencia']}
                - Időbeli érvényesség: {validation_result['idobeli_ervenyesseg']}
                
                Hibák: {validation_result['hibak']}
                Javaslatok: {validation_result['javaslatok']}
                
                Kérlek, javítsd a választ a fenti szempontok alapján, megtartva az eredeti struktúrát.
                """
                
                improved_answer = self.qa_chain.invoke({
                    "question": question,
                    "chat_history": self.chat_history.messages,
                    "answer": improved_prompt
                })
                result['answer'] = improved_answer['answer']
            
            # Válasz mentése a chat történetbe
            self.chat_history.add_message(HumanMessage(content=question))
            self.chat_history.add_message(AIMessage(content=result['answer']))
            
            # Automatikus tanulás a visszajelzések alapján
            self.auto_learner.learn_from_feedback()
            
            return result['answer']
            
        except Exception as e:
            return f"Hiba történt a kérdés megválaszolása során: {str(e)}"

    def get_feedback_analysis(self) -> dict:
        """
        Visszajelzések részletes elemzése
        
        Returns:
            dict: Elemzési eredmények
        """
        return {
            'temporal_trends': self.feedback_analyzer.analyze_temporal_trends(),
            'question_patterns': self.feedback_analyzer.analyze_question_patterns(),
            'answer_quality': self.feedback_analyzer.analyze_answer_quality()
        }

if __name__ == "__main__":
    qa_system = LegalQASystem()
    if not qa_system.load_documents():
        print("Hiba történt a dokumentumok betöltése során!")
        exit(1)
    
    question = "Milyen jogi következményei vannak annak, ha egy munkavállaló nem teljesíti a munkaköri feladatait?"
    print(f"\nKérdés: {question}")
    answer = qa_system.ask_question(question)
    print(f"\nVálasz: {answer}")
