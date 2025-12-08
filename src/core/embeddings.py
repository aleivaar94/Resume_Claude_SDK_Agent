"""
OpenAI Embeddings wrapper for text-embedding-3-small model.

This module provides a simple interface to generate embeddings using OpenAI's
text-embedding-3-small model for RAG applications.

Usage
-----
from src.core.embeddings import OpenAIEmbeddings

embedder = OpenAIEmbeddings()
embeddings = embedder.embed_texts(["text1", "text2"])
query_embedding = embedder.embed_query("search query")
"""

import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv


class OpenAIEmbeddings:
    """
    Wrapper for OpenAI text-embedding-3-small model.
    
    This class provides methods to generate embeddings for text chunks and queries
    using OpenAI's embedding API. It loads the API key from environment variables.
    
    Attributes
    ----------
    client : OpenAI
        OpenAI client instance.
    model : str
        Model name (text-embedding-3-small).
    dimensions : int
        Embedding vector dimensions (1536).
    
    Examples
    --------
    >>> embedder = OpenAIEmbeddings()
    >>> texts = ["First document", "Second document"]
    >>> embeddings = embedder.embed_texts(texts)
    >>> print(len(embeddings))  # 2
    >>> print(len(embeddings[0]))  # 1536
    """
    
    def __init__(self):
        """
        Initialize OpenAI embeddings with API key from .env file.
        
        Raises
        ------
        ValueError
            If OPENAI_API_KEY is not found in environment variables.
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please add it to your .env file."
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = "text-embedding-3-small"
        self.dimensions = 1536
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts (batch processing).
        
        Parameters
        ----------
        texts : List[str]
            List of text strings to embed.
        
        Returns
        -------
        List[List[float]]
            List of embedding vectors, one per input text.
        
        Examples
        --------
        >>> embedder = OpenAIEmbeddings()
        >>> texts = ["Machine learning", "Data science", "AI research"]
        >>> embeddings = embedder.embed_texts(texts)
        >>> len(embeddings)
        3
        """
        if not texts:
            return []
        
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        
        return [data.embedding for data in response.data]
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Parameters
        ----------
        text : str
            Query text to embed.
        
        Returns
        -------
        List[float]
            Embedding vector of length 1536.
        
        Examples
        --------
        >>> embedder = OpenAIEmbeddings()
        >>> query = "Python developer with ML experience"
        >>> embedding = embedder.embed_query(query)
        >>> len(embedding)
        1536
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        
        return response.data[0].embedding
