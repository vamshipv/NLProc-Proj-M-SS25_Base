import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from typing import List, Tuple

class Retriever:
    """
    A Retriever class to perform semantic similarity search using FAISS and SentenceTransformers.
    
    Methods:
        add_documents(texts: List[str]) -> None: Adds text documents to the index.
        query(question: str, top_k: int = 3) -> List[Tuple[str, float]]: Retrieves top-k matching chunks.
        save(path: str = "retriever") -> None: Saves the FAISS index and metadata to disk.
        load(path: str = "retriever") -> None: Loads a previously saved FAISS index and metadata.
    """
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", chunk_size=100):
        """
        Initializes the Retriever with a SentenceTransformer model and FAISS index.
        
        Args:
            model_name (str): Name of the sentence-transformer model.
            chunk_size (int): Size of each text chunk.
        """
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.documents = []
        self.embeddings = None
        self.index = None
        self.id_map = []

    def _chunk_text(self, text: str) -> List[str]:
        """Splits the input text into smaller chunks of fixed size."""
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def add_documents(self, texts: List[str]) -> None:
        """
        Adds text documents to the FAISS index after chunking and embedding.
        
        Args:
            texts (List[str]): A list of document strings to be added.
        """
        for text in texts:
            chunks = self._chunk_text(text)
            self.documents.extend(chunks)
            self.id_map.extend(chunks)

        self.embeddings = self.model.encode(self.documents, convert_to_numpy=True)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def query(self, question: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieves the top-k most similar chunks for the given query.
        
        Args:
            question (str): The query string.
            top_k (int): Number of top similar results to return.
            
        Returns:
            List[Tuple[str, float]]: List of (text_chunk, similarity_score) tuples.
        """
        query_vec = self.model.encode([question], convert_to_numpy=True)
        distances, indices = self.index.search(query_vec, top_k)
        return [(self.id_map[i], float(d)) for i, d in zip(indices[0], distances[0])]

    def save(self, path: str = "retriever") -> None:
        """
        Saves the FAISS index and metadata to disk.
        
        Args:
            path (str): Directory to save the model and index.
        """
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump({'id_map': self.id_map, 'chunk_size': self.chunk_size}, f)

    def load(self, path: str = "retriever") -> None:
        """
        Loads the FAISS index and metadata from disk.
        
        Args:
            path (str): Directory from which to load the model and index.
        """
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            metadata = pickle.load(f)
            self.id_map = metadata['id_map']
            self.chunk_size = metadata['chunk_size']


##feeding real documents
if __name__ == "__main__":
    def load_text_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    from PyPDF2 import PdfReader

    def load_pdf(file_path):
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
            return text

    # Example usage:
    text1 = load_text_file("Textdoc-1.txt")
    text2 = load_text_file("Textdoc-2.txt")
    text3 = load_text_file("Textdoc-3.txt")
    text4 = load_text_file("Textdoc-4.txt")
    pdf_text1 = load_pdf("PDFDoc-1.pdf")
    pdf_text2 = load_pdf("PDFDoc-2.pdf")

    retriever = Retriever()
    retriever.add_documents([text1, text2, text3, text4, pdf_text1, pdf_text2])
    #retriever.add_documents([text])
    retriever.save()

    # Querying across all documents
    queries = [
        "Which character is loved by children?", #Winnie the pooh text 1
        "When did the Industrial Revolution begin?",  # Expected from text2 (history)
        "What is photosynthesis?",  # Expected from text3 (biology)
        "How is AI transforming industries?",  # Expected from text4 (technology)
        "What requires more time and focus?", # Expected from pdf_text1
        "Which planet is third from the Sun?",  # Expected from pdf_text2 (space)
    ]

    for query in queries:
        print(f"Query: {query}")
        results = retriever.query(query, top_k=2)  # Fetch top 2 most similar chunks
        for text, score in results:
            print(f"Chunk: {text} | Score: {score}")
        print("-" * 50)

    retriever = Retriever()
    retriever.add_documents(["This is a sample text for testing purposes."])
    print(retriever.query("What is this text about?", top_k=1))
