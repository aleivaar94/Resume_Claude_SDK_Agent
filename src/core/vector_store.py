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
import time
import shutil
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
        
        # Store storage path for client reinitialization
        self.storage_path = storage_path
        
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
        
        # Ensure collection exists before adding documents
        if not self.client.collection_exists(self.collection_name):
            print(f"ℹ️  Collection doesn't exist. Creating it now...")
            self._create_collection_if_not_exists()
        
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
        Reinitializes the client to clear stale cached data.
        """
        # Check if collection exists first
        if not self.client.collection_exists(self.collection_name):
            print(f"ℹ️  Collection '{self.collection_name}' does not exist")
            return
        
        # Delete the collection
        self.client.delete_collection(collection_name=self.collection_name)
        print(f"🗑️  Deleted collection '{self.collection_name}'")
        
        # CRITICAL: Reinitialize client to clear stale cached data
        # QdrantClient caches collection metadata - reusing same instance
        # after delete_collection() returns stale data from the old collection
        del self.client
        import gc
        gc.collect()
        time.sleep(1.5)  # Wait for disk I/O and garbage collection
        
        # Create new client instance with clean cache
        self.client = QdrantClient(path=self.storage_path)
        print(f"🔄 Reinitialized client (cleared cache)")
        
        # Recreate empty collection
        self._create_collection_if_not_exists()
        
        # Verify deletion completed successfully
        max_retries = 3
        for attempt in range(max_retries):
            try:
                final_count = self.count_documents()
                if final_count == 0:
                    print(f"✅ Verified collection is empty ({final_count} documents)")
                    break
                else:
                    print(f"⚠️  Warning: {final_count} documents still exist after deletion attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait longer
            except Exception as e:
                print(f"⚠️  Could not verify count (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
        else:
            print(f"❌ Failed to fully clear collection after {max_retries} attempts. Use reset_database() method instead.")
    
    def count_documents(self) -> int:
        """
        Get the total number of documents in the collection.
        
        Returns
        -------
        int
            Number of documents stored. Returns 0 if collection doesn't exist.
        """
        try:
            if not self.client.collection_exists(self.collection_name):
                return 0
            
            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count
        except Exception as e:
            print(f"⚠️  Error counting documents: {e}")
            return 0
    
    def reset_database(self) -> None:
        """
        Completely reset the vector database by deleting storage folder.
        
        This is a more aggressive approach that ensures all data is cleared.
        Use when delete_collection() has race condition issues.
        
        Notes
        -----
        This method:
        1. Deletes the collection (clean shutdown)
        2. Closes the client to release file handles
        3. Removes the entire storage folder from disk
        4. Recreates the storage directory
        5. Reinitializes the client
        6. Creates a fresh empty collection
        
        Examples
        --------
        >>> store = QdrantVectorStore()
        >>> store.reset_database()
        🗑️  Deleted collection 'resume_data'
        🔌 Closed client connection
        🗑️  Deleted storage folder: ./vector_db/qdrant_storage
        📁 Recreated storage folder
        ✅ Database completely reset
        """
        try:
            # Delete collection first (clean shutdown)
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(collection_name=self.collection_name)
                print(f"🗑️  Deleted collection '{self.collection_name}'")
            
            # Get storage path from instance variable
            storage_path = Path(self.storage_path)
            
            # CRITICAL for Windows: Force close client to release file handles
            # Delete the client object and force garbage collection
            del self.client
            print(f"🔌 Closed client connection")
            
            # Force Python garbage collection to release file handles
            import gc
            gc.collect()
            
            # Wait for OS to fully release file handles (Windows needs more time)
            time.sleep(2)
            
            if storage_path.exists():
                try:
                    # Try to delete storage folder
                    shutil.rmtree(storage_path)
                    print(f"🗑️  Deleted storage folder: {storage_path}")
                except PermissionError as e:
                    # If still locked, wait longer and retry
                    print(f"⚠️  Storage locked, waiting 3 more seconds...")
                    time.sleep(3)
                    gc.collect()
                    shutil.rmtree(storage_path)
                    print(f"🗑️  Deleted storage folder: {storage_path}")
                
                time.sleep(1)  # Wait for OS cleanup
                
                # Recreate storage directory
                storage_path.mkdir(parents=True, exist_ok=True)
                print(f"📁 Recreated storage folder")
            
            # Reinitialize client with clean storage
            self.client = QdrantClient(path=str(storage_path))
            
            # Create fresh collection
            self._create_collection_if_not_exists()
            
            print(f"✅ Database completely reset")
            
        except Exception as e:
            print(f"❌ Error during database reset: {e}")
            raise
