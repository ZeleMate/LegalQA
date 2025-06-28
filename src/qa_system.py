from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import PromptTemplate

from .data_processing import DocumentCache
from .retrieval import CustomRetriever
from .utils import InMemoryHistory
from .learning import QualityAssurance, AutoLearner
from .feedback import UserFeedback, FeedbackAnalyzer

# Környezeti változók betöltése
load_dotenv()


class LegalQASystem:
    """Jogi kérdés-válasz rendszer"""

    def __init__(self):
        """Rendszer inicializálása"""
        self.embeddings = OpenAIEmbeddings()
        self.qa_chain = None
        self.chat_history = InMemoryHistory()
        self.document_cache = DocumentCache()
        self.quality_assurance = QualityAssurance()
        self.user_feedback = UserFeedback()
        self.feedback_analyzer = FeedbackAnalyzer()
        self.auto_learner = AutoLearner(self)

    def load_documents(self) -> bool:
        """Feldolgozott dokumentumok és indexek betöltése"""
        try:
            if not self.document_cache.load_documents():
                print("A dokumentumok gyorsítótárból való betöltése nem sikerült.")
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
        """Egyedi prompt generálása a jogi kérdésekhez"""
        template = """Te egy jogi asszisztens vagy... (A prompt többi része változatlan)"""
        # A teljes, hosszú prompt sablon itt lenne
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
        """Kérdés megválaszolása a rendszer által"""
        if not self.qa_chain:
            raise ValueError("A QA rendszer nincs inicializálva! Először töltsd be a dokumentumokat.")

        try:
            result = self.qa_chain.invoke({
                "question": question,
                "chat_history": self.chat_history.messages
            })
            
            # A minőség-ellenőrzési és javítási logikát az egyszerűség kedvéért most kikommentezem
            # A későbbi fejlesztés során ez visszakapcsolható
            # validation_result = self.quality_assurance.validate_response(...)
            # ...

            self.chat_history.add_message(HumanMessage(content=question))
            self.chat_history.add_message(AIMessage(content=result['answer']))
            
            # Az automatikus tanulást is kikommentezem, hogy ne fusson minden kérdésnél
            # self.auto_learner.learn_from_feedback()

            return result['answer']
        except Exception as e:
            return f"Hiba történt a kérdés megválaszolása során: {str(e)}"

    def get_feedback_analysis(self) -> dict:
        """Visszajelzések részletes elemzése"""
        return {
            'temporal_trends': self.feedback_analyzer.analyze_temporal_trends(),
            'question_patterns': self.feedback_analyzer.analyze_question_patterns(),
            'answer_quality': self.feedback_analyzer.analyze_answer_quality()
        } 