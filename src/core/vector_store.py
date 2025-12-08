"""
Qdrant vector store manager for local persistent storage.

This module provides a wrapper around Qdrant client for managing embeddings,
collections, and semantic search operations.

Usage
-----
from src.core.vector_store import QdrantVectorStore

store = QdrantVectorStore()
store.add_documents(chunks, embeddings)
results = store.search(query_vector, top_k=10, section_filter="work_experience")
"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
import uuid


class QdrantVectorStore:
    """
    Qdrant vector database manager for resume RAG system.
    
    This class handles all vector store operations including collection management,
    document insertion, and semantic search with metadata filtering.
    
    Attributes
    ----------
    client : QdrantClient
        Qdrant client instance with local persistent storage.
    collection_name : str
        Name of the collection storing resume documents.
    vector_size : int
        Dimension of embedding vectors (1536 for text-embedding-3-small).
    
    Examples
    --------
    >>> store = QdrantVectorStore()
    >>> chunks = [
    ...     {
    ...         "content": "Built ETL pipeline...",
    ...         "source_file": "resume_ale.md",
    ...         "section_type": "work_experience",
    ...         "metadata": {"company": "CFIA", "position": "Data Scientist"}
    ...     }
    ... ]
    >>> embeddings = [[0.1, 0.2, ...], ...]
    >>> store.add_documents(chunks, embeddings)
    """
    
    def __init__(self, storage_path: str = "./vector_db/qdrant_storage"):
        """
        Initialize Qdrant vector store with local persistent storage.
        
        Parameters
        ----------
        storage_path : str, optional
            Path to local Qdrant storage directory (default: ./vector_db/qdrant_storage).
        
        Notes
        -----
        Creates storage directory if it doesn't exist.
        Automatically creates collection if it doesn't exist.
        """
        # Create storage directory
        Path(storage_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize Qdrant client with local storage
        self.client = QdrantClient(path=storage_path)
        self.collection_name = "resume_data"
        self.vector_size = 1536  # text-embedding-3-small dimension
        
        # Create collection if it doesn't exist
        self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self) -> None:
        """
        Create collection with appropriate vector configuration if it doesn't exist.
        
        Notes
        -----
        Uses cosine distance metric for similarity search.
        Vector size is set to 1536 for OpenAI text-embedding-3-small.
        """
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created collection '{self.collection_name}'")
        else:
            print(f"ℹ️  Collection '{self.collection_name}' already exists")
    
    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> None:
        """
        Add document chunks with embeddings to the vector store.
        
        Parameters
        ----------
        chunks : List[Dict[str, Any]]
            List of document chunks with structure:
            {
                "content": str,
                "source_file": str,
                "section_type": str,
                "metadata": dict
            }
        embeddings : List[List[float]]
            List of embedding vectors corresponding to chunks.
        
        Raises
        ------
        ValueError
            If chunks and embeddings lists have different lengths.
        
        Examples
        --------
        >>> chunks = [
        ...     {
        ...         "content": "Achievement text...",
        ...         "source_file": "resume_ale.md",
        ...         "section_type": "work_experience",
        ...         "metadata": {
        ...             "company": "CFIA",
        ...             "position": "Data Scientist II",
        ...             "start_date": "March-2025",
        ...             "end_date": "November-2025",
        ...             "achievement_index": 0
        ...         }
        ...     }
        ... ]
        >>> embeddings = [[0.1, 0.2, ...]]  # 1536-dim vectors
        >>> store.add_documents(chunks, embeddings)
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )
        
        # Create points for batch upload
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "content": chunk["content"],
                    "source_file": chunk["source_file"],
                    "section_type": chunk["section_type"],
                    "metadata": chunk.get("metadata", {})
                }
            )
            points.append(point)
        
        # Batch upload to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"✅ Added {len(points)} documents to '{self.collection_name}'")
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        section_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search in the vector store.
        
        Parameters
        ----------
        query_vector : List[float]
            Query embedding vector.
        top_k : int, optional
            Number of results to return (default: 10).
        section_filter : str, optional
            Filter by section_type (e.g., "work_experience", "education").
        
        Returns
        -------
        List[Dict[str, Any]]
            List of search results with structure:
            {
                "id": str,
                "score": float,
                "content": str,
                "source_file": str,
                "section_type": str,
                "metadata": dict
            }
        
        Examples
        --------
        >>> query_vector = embedder.embed_query("Python ETL pipelines")
        >>> results = store.search(query_vector, top_k=5, section_filter="work_experience")
        >>> for result in results:
        ...     print(f"Score: {result['score']:.3f}")
        ...     print(f"Company: {result['metadata']['company']}")
        """
        # Build filter if section_filter is provided
        search_filter = None
        if section_filter:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="section_type",
                        match=MatchValue(value=section_filter)
                    )
                ]
            )
        
        # Perform search using query_points (updated API)
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=search_filter
        )
        
        # Format results - query_points returns a QueryResponse with points
        results = []
        for result in search_results.points:
            results.append({
                "id": result.id,
                "score": result.score,
                "content": result.payload["content"],
                "source_file": result.payload["source_file"],
                "section_type": result.payload["section_type"],
                "metadata": result.payload.get("metadata", {})
            })
        
        return results
    
    def delete_collection(self) -> None:
        """
        Delete the collection (useful for resetting/rebuilding).
        
        Notes
        -----
        This permanently deletes all documents in the collection.
        A new collection will be created on next add_documents call.
        """
        self.client.delete_collection(collection_name=self.collection_name)
        print(f"🗑️  Deleted collection '{self.collection_name}'")
        
        # Recreate empty collection
        self._create_collection_if_not_exists()
    
    def count_documents(self) -> int:
        """
        Get the total number of documents in the collection.
        
        Returns
        -------
        int
            Number of documents stored.
        """
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.points_count
