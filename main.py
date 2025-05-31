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
    embeddings: OpenAIEmbeddings = Field(..., description="OpenAI embeddings model")
    faiss_index: faiss.Index = Field(..., description="FAISS index")
    id_mapping: dict = Field(..., description="ID mapping dictionary")
    documents_df: pd.DataFrame = Field(..., description="Documents DataFrame")
    graph: nx.Graph = Field(..., description="Graph data")

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        # Kérdezés beágyazása
        query_embedding = self.embeddings.embed_query(query)
        
        # FAISS keresés - numpy array konvertálás
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # FAISS keresés - több releváns dokumentum (5 helyett 8)
        D, I = self.faiss_index.search(query_embedding, 8)
        
        # Dokumentumok lekérése és relevancia alapján rendezése
        documents = []
        seen_doc_ids = set()  # Duplikált dokumentumok elkerülése
        
        for idx, distance in zip(I[0], D[0]):
            if idx in self.id_mapping:
                doc_id = self.id_mapping[idx]
                if doc_id in seen_doc_ids:
                    continue
                    
                matching_docs = self.documents_df[self.documents_df['doc_id'] == doc_id]
                if not matching_docs.empty:
                    doc = matching_docs.iloc[0]
                    
                    # Metaadatok összeállítása
                    metadata = {
                        'MeghozoBirosag': doc['MeghozoBirosag'],
                        'JogTerulet': doc['JogTerulet'],
                        'Jogszabalyhelyek': doc['Jogszabalyhelyek'],
                        'HatarozatEve': doc['HatarozatEve'],
                        'doc_id': doc_id,
                        'relevancia': float(1 / (1 + distance))  # Relevancia pontszám
                    }
                    documents.append(Document(
                        page_content=f"Dokumentum azonosító: {doc_id}",  # Csak az azonosítót adjuk vissza
                        metadata=metadata
                    ))
                    seen_doc_ids.add(doc_id)
        
        # Gráf alapú relevancia növelése - több szomszéd
        if self.graph and len(documents) > 0:
            for doc in documents:
                if hasattr(doc, 'metadata') and doc.metadata and isinstance(doc.metadata, dict) and 'doc_id' in doc.metadata:
                    doc_id = doc.metadata['doc_id']
                    if doc_id in self.graph:
                        neighbors = list(self.graph.neighbors(doc_id))[:3]  # 3 szomszéd
                        for neighbor in neighbors:
                            if neighbor not in seen_doc_ids and len(documents) < 12:  # Maximum 12 dokumentum összesen
                                neighbor_doc = self.documents_df[self.documents_df['doc_id'] == neighbor]
                                if not neighbor_doc.empty:
                                    neighbor_row = neighbor_doc.iloc[0]
                                    
                                    neighbor_metadata = {
                                        'MeghozoBirosag': neighbor_row['MeghozoBirosag'],
                                        'JogTerulet': neighbor_row['JogTerulet'],
                                        'Jogszabalyhelyek': neighbor_row['Jogszabalyhelyek'],
                                        'HatarozatEve': neighbor_row['HatarozatEve'],
                                        'doc_id': neighbor,
                                        'relevancia': 0.7  # Alacsonyabb relevancia a szomszédoknak
                                    }
                                    documents.append(Document(
                                        page_content=f"Dokumentum azonosító: {neighbor}",  # Csak az azonosítót adjuk vissza
                                        metadata=neighbor_metadata
                                    ))
                                    seen_doc_ids.add(neighbor)
        
        # Relevancia szerint rendezés
        documents.sort(key=lambda x: x.metadata.get('relevancia', 0), reverse=True)
        
        return documents

class LegalQASystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.qa_chain = None
        self.chat_history = InMemoryHistory()
        self.documents_df = None
        self.faiss_index = None
        self.id_mapping = None
        self.graph = None

    def load_documents(self):
        """Feldolgozott dokumentumok és indexek betöltése"""
        try:
            # Parquet fájl betöltése
            self.documents_df = pd.read_parquet('processed_data/processed_documents_with_embeddings.parquet')
            
            # DEBUG: oszlopnevek kiírása
            print("Oszlopnevek a DataFrame-ben:", self.documents_df.columns.tolist())
            # DEBUG: első 3 sor kiírása
            print("Első 3 sor az adathalmazból:")
            print(self.documents_df.head(3))
            print("Első sor típusa:", type(self.documents_df.iloc[0]))
            
            # FAISS index betöltése
            self.faiss_index = faiss.read_index('processed_data/faiss_index.bin')
            
            # ID leképezések betöltése
            with open('processed_data/faiss_id_mapping.pkl', 'rb') as f:
                self.id_mapping = pickle.load(f)
            
            # Gráf betöltése
            self.graph = nx.read_graphml('processed_data/graph_data/graph.graphml')
            
            self._initialize_qa_chain()
            return True
        except Exception as e:
            print(f"Hiba az adatok betöltése során: {str(e)}")
            return False

    def _initialize_qa_chain(self):
        """QA lánc inicializálása"""
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
        """Egyedi prompt a jogi kérdésekhez"""
        from langchain.prompts import PromptTemplate
        
        template = """Te egy jogi asszisztens vagy, aki a megadott jogi dokumentumok alapján válaszol a kérdésekre.
        
        Fontos szabályok:
        1. CSAK a megadott kontextusban található dokumentumok azonosítóit listázd fel
        2. Minden dokumentum esetén add meg a következő információkat:
           - Dokumentum azonosító (doc_id)
           - Meghozó bíróság
           - Jogtérület
           - Határozat éve
        3. Ha nincs releváns dokumentum, azt jelezd: "Nem található releváns dokumentum ehhez a kérdéshez."
        4. A dokumentumokat relevancia szerint rendezd (csökkenő sorrendben)
        
        Kontextus: {context}
        
        Kérdés: {question}
        
        Válasz:"""
        
        return PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

    def ask_question(self, question):
        """Kérdés megválaszolása"""
        if not self.qa_chain:
            raise ValueError("A QA rendszer nincs inicializálva! Először töltsd be a dokumentumokat.")
        
        try:
            result = self.qa_chain.invoke({
                "question": question,
                "chat_history": self.chat_history.messages
            })
            return result['answer']
        except Exception as e:
            return f"Hiba történt a kérdés megválaszolása során: {str(e)}"

def main():
    # Rendszer inicializálása
    qa_system = LegalQASystem()
    if not qa_system.load_documents():
        print("Hiba történt a dokumentumok betöltése során!")
        return
    
    # Teszt kérdés
    question = "Milyen jogi következményei vannak annak, ha egy munkavállaló nem teljesíti a munkaköri feladatait?"
    print(f"\nKérdés: {question}")
    answer = qa_system.ask_question(question)
    print(f"\nVálasz: {answer}")

if __name__ == "__main__":
    main()
