import os
import faiss
import pickle
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from app import db
from models import Document, Chunk, ChunkEmbedding, EmbeddingModel, Metadata
from embedding_models import EmbeddingModelManager

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages vector indices for semantic search using FAISS
    """
    
    def __init__(self, embedding_manager: EmbeddingModelManager):
        """
        Initialize vector store
        
        Args:
            embedding_manager: Instance of EmbeddingModelManager
        """
        self.embedding_manager = embedding_manager
        self.indices = {}  # {model_id: faiss_index}
        self.chunk_ids = {}  # {model_id: [chunk_ids]}
        self._load_indices()
    
    def _load_indices(self) -> None:
        """Load vector indices for all active models"""
        active_models = self.embedding_manager.get_active_models()
        
        for model_info in active_models:
            self._load_or_create_index(model_info)
    
    def _load_or_create_index(self, model_info: EmbeddingModel) -> Tuple[faiss.Index, List[str]]:
        """
        Load or create a FAISS index for a specific model
        
        Args:
            model_info: EmbeddingModel database object
            
        Returns:
            Tuple of (faiss_index, chunk_ids)
        """
        if model_info.id in self.indices:
            return self.indices[model_info.id], self.chunk_ids[model_info.id]
        
        # Get all embeddings for this model
        embeddings_query = db.session.query(
            ChunkEmbedding.embedding_data,
            ChunkEmbedding.chunk_id
        ).filter(
            ChunkEmbedding.model_id == model_info.id
        ).all()
        
        # If no embeddings exist yet, create an empty index
        if not embeddings_query:
            logger.info(f"Creating empty FAISS index for model {model_info.name}")
            index = faiss.IndexFlatIP(model_info.dimension)  # Inner product similarity
            self.indices[model_info.id] = index
            self.chunk_ids[model_info.id] = []
            return index, []
        
        # Extract embeddings and chunk IDs
        embeddings = []
        chunk_ids = []
        
        for embedding_data, chunk_id in embeddings_query:
            vector = pickle.loads(embedding_data)
            embeddings.append(vector)
            chunk_ids.append(chunk_id)
        
        # Create and populate FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        index = faiss.IndexFlatIP(model_info.dimension)
        index.add(embeddings_array)
        
        self.indices[model_info.id] = index
        self.chunk_ids[model_info.id] = chunk_ids
        
        logger.info(f"Loaded FAISS index for model {model_info.name} with {len(chunk_ids)} vectors")
        return index, chunk_ids
    
    def add_embedding(self, chunk_id: str, model_id: str, embedding_vector: np.ndarray) -> None:
        """
        Add an embedding to the FAISS index
        
        Args:
            chunk_id: ID of the chunk
            model_id: ID of the model
            embedding_vector: Numpy array with embedding vector
        """
        # Get the index for this model
        model_info = EmbeddingModel.query.get(model_id)
        if not model_info:
            raise ValueError(f"Model with ID {model_id} not found")
        
        index, chunk_ids = self._load_or_create_index(model_info)
        
        # Add embedding to index
        embeddings_array = np.array([embedding_vector]).astype('float32')
        index.add(embeddings_array)
        
        # Add chunk ID to list
        chunk_ids.append(chunk_id)
        
        logger.debug(f"Added embedding for chunk {chunk_id} to index for model {model_info.name}")
    
    def update_embedding(self, chunk_id: str, model_id: str, embedding_vector: np.ndarray) -> None:
        """
        Update an embedding in the FAISS index
        
        Args:
            chunk_id: ID of the chunk
            model_id: ID of the model
            embedding_vector: Numpy array with embedding vector
        """
        # FAISS doesn't support direct updates, so we need to rebuild the index
        self._rebuild_index(model_id)
    
    def delete_embedding(self, chunk_id: str, model_id: str) -> None:
        """
        Delete an embedding from the FAISS index
        
        Args:
            chunk_id: ID of the chunk
            model_id: ID of the model
        """
        # FAISS doesn't support direct deletion, so we need to rebuild the index
        self._rebuild_index(model_id)
    
    def _rebuild_index(self, model_id: str) -> None:
        """
        Rebuild the FAISS index for a specific model
        
        Args:
            model_id: ID of the model
        """
        # Remove existing index
        if model_id in self.indices:
            del self.indices[model_id]
        if model_id in self.chunk_ids:
            del self.chunk_ids[model_id]
        
        # Reload index
        model_info = EmbeddingModel.query.get(model_id)
        if model_info:
            self._load_or_create_index(model_info)
            logger.info(f"Rebuilt FAISS index for model {model_info.name}")
    
    def search(self, 
              query_text: str, 
              model_id: Optional[str] = None, 
              top_k: int = 5, 
              min_score: float = 0.5,
              metadata_filters: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Perform semantic search with a text query
        
        Args:
            query_text: Text query for search
            model_id: ID of model to use (uses default if None)
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            metadata_filters: Optional metadata filters {key: value}
            
        Returns:
            List of result dictionaries with chunk, document, and score information
        """
        # Get model to use
        if not model_id:
            model_info = self.embedding_manager.get_default_model()
            if not model_info:
                active_models = self.embedding_manager.get_active_models()
                if not active_models:
                    raise ValueError("No active embedding models available")
                model_info = active_models[0]
            model_id = model_info.id
        else:
            model_info = EmbeddingModel.query.get(model_id)
            if not model_info:
                raise ValueError(f"Model with ID {model_id} not found")
            if not model_info.is_active:
                raise ValueError(f"Model {model_info.name} is not active")
        
        # Get or load the model
        model = self.embedding_manager._load_model(model_info)
        
        # Generate query embedding
        query_vector = model.encode(query_text)
        query_vector = np.array([query_vector]).astype('float32')
        
        # Get the index for this model
        index, chunk_ids = self._load_or_create_index(model_info)
        
        # Search the index
        scores, indices = index.search(query_vector, min(top_k * 2, len(chunk_ids)))  # Get more results for filtering
        
        # Filter and format results
        results = []
        added_document_ids = set()  # Track which documents we've already added
        
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(chunk_ids) or scores[0][i] < min_score:
                continue
            
            chunk_id = chunk_ids[idx]
            chunk = Chunk.query.get(chunk_id)
            if not chunk:
                continue
            
            document = Document.query.get(chunk.document_id)
            if not document:
                continue
            
            # Apply metadata filters if specified
            if metadata_filters:
                metadata_match = self._check_metadata_filters(document.id, metadata_filters)
                if not metadata_match:
                    continue
            
            # Skip if we already have a result from this document and have enough results
            if document.id in added_document_ids and len(results) >= top_k:
                continue
            
            # Add document ID to tracking set
            added_document_ids.add(document.id)
            
            # Get document metadata
            metadata = {m.key: m.value for m in Metadata.query.filter_by(document_id=document.id).all()}
            
            # Add result
            results.append({
                'chunk': {
                    'id': chunk.id,
                    'content': chunk.content,
                    'index': chunk.chunk_index,
                    'start_pos': chunk.start_pos,
                    'end_pos': chunk.end_pos
                },
                'document': {
                    'id': document.id,
                    'name': document.name,
                    'mime_type': document.mime_type,
                    'created_at': document.created_at.isoformat(),
                    'updated_at': document.updated_at.isoformat(),
                    'metadata': metadata
                },
                'score': float(scores[0][i])
            })
            
            # Stop if we have enough results
            if len(results) >= top_k:
                break
        
        return results
    
    def _check_metadata_filters(self, document_id: str, filters: Dict[str, str]) -> bool:
        """
        Check if a document matches metadata filters
        
        Args:
            document_id: Document ID to check
            filters: Metadata filters {key: value}
            
        Returns:
            True if document matches all filters
        """
        for key, value in filters.items():
            metadata = Metadata.query.filter_by(document_id=document_id, key=key, value=value).first()
            if not metadata:
                return False
        return True
    
    def sync_document_embeddings(self, document_id: str, model_id: Optional[str] = None) -> int:
        """
        Synchronize embeddings for a document
        
        Args:
            document_id: Document ID
            model_id: Optional model ID (syncs for all active models if None)
            
        Returns:
            Number of embeddings updated
        """
        count = 0
        
        # Get all chunks for this document
        chunks = Chunk.query.filter_by(document_id=document_id).all()
        
        # Get models to use
        if model_id:
            models = [EmbeddingModel.query.get(model_id)]
            if not models[0]:
                raise ValueError(f"Model with ID {model_id} not found")
            if not models[0].is_active:
                raise ValueError(f"Model {models[0].name} is not active")
        else:
            models = self.embedding_manager.get_active_models()
        
        # Generate embeddings for each chunk and model
        for chunk in chunks:
            for model_info in models:
                # Generate and store embedding
                model_id, embedding_vector = self.embedding_manager.generate_embeddings(chunk, model_info.id)
                
                # Update index
                self._rebuild_index(model_id)
                count += 1
        
        logger.info(f"Synchronized {count} embeddings for document {document_id}")
        return count
