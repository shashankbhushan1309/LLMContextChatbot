import torch
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Union

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model = SentenceTransformer(model_name)
        self.embedding_dimension = self.model.get_sentence_embedding_dimension()
    
    def generate_embeddings(self, texts: Union[str, List[str]]):
        """
        Generate embeddings for input text(s)
        
        Args:
            texts: String or list of strings to embed
            
        Returns:
            Embeddings in format compatible with ChromaDB
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Generate embeddings as numpy arrays (not tensors)
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # If it's a single text, return a single embedding
        if len(texts) == 1:
            return embeddings[0].tolist()
            
        # Return as list of lists for multiple texts
        return embeddings.tolist()
    
    def process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of text chunks and add embeddings
        
        Args:
            chunks: List of chunk dictionaries with 'text' key
            
        Returns:
            Chunks with added embeddings
        """
        if not chunks:
            return []
            
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.generate_embeddings(texts)
        
        # If single chunk, embeddings is a single list
        if len(chunks) == 1:
            chunks[0]["embedding"] = embeddings
        else:
            # Otherwise embeddings is a list of lists
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i]
            
        return chunks 