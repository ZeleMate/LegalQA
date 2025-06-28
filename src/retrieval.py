import faiss
import numpy as np
import pandas as pd
import networkx as nx
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class CustomRetriever(BaseRetriever, BaseModel):
    """Egyedi dokumentum kereső a jogi dokumentumokhoz"""

    embeddings: OpenAIEmbeddings = Field(..., description="OpenAI embeddings model")
    faiss_index: faiss.Index = Field(..., description="FAISS index")
    id_mapping: dict = Field(..., description="ID mapping dictionary")
    documents_df: pd.DataFrame = Field(..., description="Documents DataFrame")
    graph: nx.Graph = Field(..., description="Graph data")

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        """
        Releváns dokumentumok keresése a kérdés alapján
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
                        'MeghozoBirosag': doc.get('MeghozoBirosag'),
                        'JogTerulet': doc.get('JogTerulet'),
                        'Jogszabalyhelyek': doc.get('Jogszabalyhelyek'),
                        'HatarozatEve': doc.get('HatarozatEve'),
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
                if 'doc_id' in doc.metadata:
                    doc_id = doc.metadata['doc_id']
                    if doc_id in self.graph:
                        neighbors = list(self.graph.neighbors(doc_id))[:3]
                        for neighbor in neighbors:
                            if neighbor not in seen_doc_ids and len(documents) < 12:
                                neighbor_doc = self.documents_df[self.documents_df['doc_id'] == neighbor]
                                if not neighbor_doc.empty:
                                    neighbor_row = neighbor_doc.iloc[0]
                                    neighbor_metadata = {
                                        'MeghozoBirosag': neighbor_row.get('MeghozoBirosag'),
                                        'JogTerulet': neighbor_row.get('JogTerulet'),
                                        'Jogszabalyhelyek': neighbor_row.get('Jogszabalyhelyek'),
                                        'HatarozatEve': neighbor_row.get('HatarozatEve'),
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