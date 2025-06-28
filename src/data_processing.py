import os
import pandas as pd
import faiss
import pickle
import networkx as nx


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
        """Dokumentumok és indexek betöltése"""
        if self.documents_df is not None:
            return True

        try:
            self.documents_df = pd.read_parquet('processed_data/processed_documents_with_embeddings.parquet')
            with open('processed_data/faiss_id_mapping.pkl', 'rb') as f:
                self.id_mapping = pickle.load(f)
            self.faiss_index = faiss.read_index('processed_data/faiss_index.bin')

            graph_path = 'processed_data/graph_data/document_graph.gpickle'
            if os.path.exists(graph_path):
                with open(graph_path, 'rb') as f:
                    self.graph = pickle.load(f)
            else:
                self.graph = nx.Graph()

            print("Dokumentumok sikeresen betöltve a gyorsítótárból.")
            return True
        except Exception as e:
            print(f"Hiba a dokumentumok betöltése közben: {str(e)}")
            return False

    def get_documents(self):
        """Dokumentumok, index és gráf lekérdezése"""
        return self.documents_df, self.faiss_index, self.id_mapping, self.graph 