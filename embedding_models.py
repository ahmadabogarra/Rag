import os
import logging
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
# Using numpy for simple embeddings instead of sentence-transformers
from app import db
from models import EmbeddingModel, ChunkEmbedding, Chunk

logger = logging.getLogger(__name__)

class EmbeddingModelManager:
    """
    Manages embedding models and generates embeddings for document chunks
    """
    
    def __init__(self):
        """Initialize the embedding model manager"""
        self.models = {}  # Cache for loaded models
        self._load_active_models()
    
    def _load_active_models(self) -> None:
        """Load all active embedding models from the database"""
        active_models = EmbeddingModel.query.filter_by(is_active=True).all()
        
        for model_info in active_models:
            self._load_model(model_info)
    
    def _load_model(self, model_info: EmbeddingModel) -> Any:
        """
        Load a specific embedding model
        
        Args:
            model_info: EmbeddingModel database object
            
        Returns:
            Loaded model object
        """
        if model_info.id in self.models:
            return self.models[model_info.id]
        
        logger.info(f"Loading model: {model_info.name} ({model_info.model_type})")
        
        try:
            # Simple embedding model using character counts and hashing
            # This is a placeholder for the actual transformer models
            class SimpleEmbedder:
                def __init__(self, dimension):
                    self.dimension = dimension
                
                def encode(self, text):
                    # Create a simple deterministic embedding based on the text
                    # This is NOT suitable for semantic search but allows the system to run
                    if not text:
                        return np.zeros(self.dimension)
                    
                    # Use character frequencies and positions for a basic embedding
                    text = text.lower()
                    chars = list(set(text))
                    
                    # Initialize the embedding vector
                    embedding = np.zeros(self.dimension)
                    
                    for i, char in enumerate(text):
                        # Use character code and position to influence different dimensions
                        char_code = ord(char)
                        pos = i % self.dimension
                        embedding[pos] += char_code / 1000
                    
                    # Normalize the embedding
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                    
                    return embedding
            
            # Create the simple embedder
            model = SimpleEmbedder(model_info.dimension)
            self.models[model_info.id] = model
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_info.name}: {str(e)}")
            raise
    
    def add_model(self, name: str, model_type: str, model_path: str, 
                  dimension: int, is_active: bool = True, 
                  is_default: bool = False, config: Optional[Dict] = None) -> EmbeddingModel:
        """
        Add a new embedding model to the database
        
        Args:
            name: Model name
            model_type: Type of model (e.g., 'sentence-transformers')
            model_path: Path or identifier for the model
            dimension: Embedding vector dimension
            is_active: Whether the model should be active
            is_default: Whether this should be the default model
            config: Additional configuration parameters
            
        Returns:
            Created EmbeddingModel object
        """
        # If setting as default, unset any existing defaults
        if is_default:
            EmbeddingModel.query.filter_by(is_default=True).update({'is_default': False})
        
        # Create new model entry
        model_info = EmbeddingModel(
            name=name,
            model_type=model_type,
            model_path=model_path,
            dimension=dimension,
            is_active=is_active,
            is_default=is_default,
            config=config
        )
        
        db.session.add(model_info)
        db.session.commit()
        
        # Load the model if it's active
        if is_active:
            self._load_model(model_info)
        
        logger.info(f"Added new model: {name} (ID: {model_info.id})")
        return model_info
    
    def update_model(self, model_id: str, **kwargs) -> EmbeddingModel:
        """
        Update an existing embedding model
        
        Args:
            model_id: ID of the model to update
            **kwargs: Fields to update
            
        Returns:
            Updated EmbeddingModel object
        """
        model_info = EmbeddingModel.query.get(model_id)
        if not model_info:
            raise ValueError(f"Model with ID {model_id} not found")
        
        reload_needed = False
        
        # If setting as default, unset any existing defaults
        if kwargs.get('is_default', False):
            EmbeddingModel.query.filter_by(is_default=True).update({'is_default': False})
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(model_info, key):
                if key in ['model_type', 'model_path', 'config'] and getattr(model_info, key) != value:
                    reload_needed = True
                setattr(model_info, key, value)
        
        db.session.commit()
        
        # Reload model if necessary
        if reload_needed and model_info.is_active:
            if model_info.id in self.models:
                del self.models[model_info.id]
            self._load_model(model_info)
        
        logger.info(f"Updated model: {model_info.name} (ID: {model_id})")
        return model_info
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete an embedding model
        
        Args:
            model_id: ID of model to delete
            
        Returns:
            True if successful
        """
        model_info = EmbeddingModel.query.get(model_id)
        if not model_info:
            raise ValueError(f"Model with ID {model_id} not found")
        
        # Remove from cache if loaded
        if model_id in self.models:
            del self.models[model_id]
        
        # Delete model from database
        db.session.delete(model_info)
        db.session.commit()
        
        logger.info(f"Deleted model: {model_info.name} (ID: {model_id})")
        return True
    
    def get_default_model(self) -> Optional[EmbeddingModel]:
        """
        Get the default embedding model
        
        Returns:
            Default EmbeddingModel object or None if no default exists
        """
        return EmbeddingModel.query.filter_by(is_default=True, is_active=True).first()
    
    def get_active_models(self) -> List[EmbeddingModel]:
        """
        Get all active embedding models
        
        Returns:
            List of active EmbeddingModel objects
        """
        return EmbeddingModel.query.filter_by(is_active=True).all()
    
    def generate_embeddings(self, chunk: Chunk, model_id: Optional[str] = None) -> Tuple[str, np.ndarray]:
        """
        Generate embeddings for a document chunk using a specific model
        
        Args:
            chunk: Chunk object to generate embeddings for
            model_id: ID of model to use (uses default if None)
            
        Returns:
            Tuple of (model_id, embedding_vector)
        """
        # Get model to use
        if not model_id:
            model_info = self.get_default_model()
            if not model_info:
                active_models = self.get_active_models()
                if not active_models:
                    raise ValueError("No active embedding models available")
                model_info = active_models[0]
        else:
            model_info = EmbeddingModel.query.get(model_id)
            if not model_info:
                raise ValueError(f"Model with ID {model_id} not found")
            if not model_info.is_active:
                raise ValueError(f"Model {model_info.name} is not active")
        
        # Load model if not already loaded
        model = self._load_model(model_info)
        
        # Generate embedding
        embedding_vector = model.encode(chunk.content)
        
        # Store embedding in database
        self._store_embedding(chunk.id, model_info.id, embedding_vector)
        
        return model_info.id, embedding_vector
    
    def _store_embedding(self, chunk_id: str, model_id: str, embedding_vector: np.ndarray) -> ChunkEmbedding:
        """
        Store a chunk embedding in the database
        
        Args:
            chunk_id: ID of the chunk
            model_id: ID of the model used
            embedding_vector: Numpy array with embedding vector
            
        Returns:
            Created or updated ChunkEmbedding object
        """
        # Check if embedding already exists
        existing = ChunkEmbedding.query.filter_by(chunk_id=chunk_id, model_id=model_id).first()
        
        # Serialize embedding vector
        embedding_data = pickle.dumps(embedding_vector)
        
        if existing:
            # Update existing embedding
            existing.embedding_data = embedding_data
            db.session.commit()
            return existing
        else:
            # Create new embedding
            embedding = ChunkEmbedding(
                chunk_id=chunk_id,
                model_id=model_id,
                embedding_data=embedding_data
            )
            db.session.add(embedding)
            db.session.commit()
            return embedding
    
    def get_embedding(self, chunk_id: str, model_id: str) -> Optional[np.ndarray]:
        """
        Get the stored embedding for a chunk and model
        
        Args:
            chunk_id: ID of the chunk
            model_id: ID of the model
            
        Returns:
            Numpy array with embedding vector or None if not found
        """
        embedding = ChunkEmbedding.query.filter_by(chunk_id=chunk_id, model_id=model_id).first()
        if not embedding:
            return None
        
        # Deserialize embedding vector
        return pickle.loads(embedding.embedding_data)
    
    def regenerate_all_embeddings(self, model_id: Optional[str] = None) -> int:
        """
        Regenerate embeddings for all chunks using a specific model
        
        Args:
            model_id: ID of model to use (regenerates for all active models if None)
            
        Returns:
            Number of embeddings generated
        """
        count = 0
        
        # Get all chunks
        chunks = Chunk.query.all()
        
        # Get models to use
        if model_id:
            models = [EmbeddingModel.query.get(model_id)]
            if not models[0]:
                raise ValueError(f"Model with ID {model_id} not found")
            if not models[0].is_active:
                raise ValueError(f"Model {models[0].name} is not active")
        else:
            models = self.get_active_models()
        
        # Generate embeddings for each chunk and model
        for chunk in chunks:
            for model_info in models:
                self.generate_embeddings(chunk, model_info.id)
                count += 1
        
        logger.info(f"Regenerated {count} embeddings")
        return count
