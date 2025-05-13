import logging
import json
from typing import Dict, Any, Optional, List, Union
import numpy as np
import faiss
from app import db
from models import EmbeddingModel
from flask import current_app

logger = logging.getLogger(__name__)

class FaissSettings:
    """
    Settings for FAISS index configuration
    """
    # Available index types
    INDEX_TYPES = {
        'flat_ip': {
            'name': 'IndexFlatIP',
            'description': 'Flat index with inner product similarity (fastest to build, slowest to search)'
        },
        'flat_l2': {
            'name': 'IndexFlatL2',
            'description': 'Flat index with L2 distance (fastest to build, slowest to search)'
        },
        'ivf_flat': {
            'name': 'IVF Flat',
            'description': 'Inverted file with flat quantizer (good balance of speed/accuracy)'
        },
        'ivf_pq': {
            'name': 'IVF PQ',
            'description': 'Inverted file with product quantization (fastest search, lower accuracy)'
        },
        'hnsw': {
            'name': 'HNSW',
            'description': 'Hierarchical Navigable Small World (high accuracy, fast search)'
        }
    }
    
    # Available metric types
    METRIC_TYPES = {
        'inner_product': {
            'name': 'Inner Product',
            'description': 'For cosine similarity with normalized vectors',
            'faiss_metric': faiss.METRIC_INNER_PRODUCT
        },
        'l2': {
            'name': 'L2 Distance',
            'description': 'Euclidean distance between vectors',
            'faiss_metric': faiss.METRIC_L2
        }
    }
    
    def __init__(self):
        """Initialize FAISS settings with defaults"""
        # Default settings
        self.index_type = 'flat_ip'
        self.metric_type = 'inner_product'
        self.embedding_dim = 384  # Will be updated based on active models
        self.index_factory_string = ''
        self.use_index_gpu = False
        
        # Number of centroids for IVF indexes - rule of thumb: sqrt(n) where n is number of vectors
        self.n_centroids = 100  
        
        # HNSW specific params
        self.hnsw_m = 16  # Number of connections per layer
        self.hnsw_ef_construction = 40  # Size of the dynamic list for the nearest neighbors (construction)
        self.hnsw_ef_search = 16  # Size of the dynamic list for the nearest neighbors (search)
        
        # PQ specific params
        self.pq_m = 8  # Number of sub-quantizers
        self.pq_bits = 8  # Number of bits per sub-quantizer
        
        # Load settings from database or file if available
        self._load_settings()
    
    def _load_settings(self) -> None:
        """Load settings from the database or config file"""
        try:
            # Try to load from application config
            if hasattr(current_app, 'config') and 'FAISS_SETTINGS' in current_app.config:
                settings = current_app.config['FAISS_SETTINGS']
                self._update_from_dict(settings)
            else:
                # Try to find the dimensions from the default embedding model
                model = EmbeddingModel.query.filter_by(is_default=True).first()
                if model:
                    self.embedding_dim = model.dimension
        except Exception as e:
            logger.warning(f"Could not load FAISS settings: {str(e)}")
    
    def _update_from_dict(self, settings: Dict[str, Any]) -> None:
        """Update settings from a dictionary"""
        for key, value in settings.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_factory_string(self) -> str:
        """
        Get the FAISS index factory string based on current settings
        
        Returns:
            FAISS index factory string
        """
        if self.index_factory_string:
            return self.index_factory_string
            
        # Build default factory string based on index type
        if self.index_type == 'flat_ip' or self.index_type == 'flat_l2':
            return "Flat"
        elif self.index_type == 'ivf_flat':
            return f"IVF{self.n_centroids},Flat"
        elif self.index_type == 'ivf_pq':
            return f"IVF{self.n_centroids},PQ{self.pq_m}x{self.pq_bits}"
        elif self.index_type == 'hnsw':
            return f"HNSW{self.hnsw_m}"
        else:
            return "Flat"  # Default to flat index
    
    def create_index(self) -> faiss.Index:
        """
        Create a FAISS index based on current settings
        
        Returns:
            FAISS index
        """
        metric = self.METRIC_TYPES[self.metric_type]['faiss_metric']
        
        # Check if we're using a complex index type that requires a factory
        if self.index_type in ['ivf_flat', 'ivf_pq', 'hnsw']:
            index = faiss.index_factory(self.embedding_dim, self.get_factory_string(), metric)
        else:
            # Simple index types
            if self.index_type == 'flat_ip':
                index = faiss.IndexFlatIP(self.embedding_dim)
            elif self.index_type == 'flat_l2':
                index = faiss.IndexFlatL2(self.embedding_dim)
            else:
                # Default to inner product
                index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Move to GPU if requested and available
        if self.use_index_gpu:
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                logger.info("FAISS index moved to GPU")
            except Exception as e:
                logger.warning(f"Could not move index to GPU: {str(e)}")
        
        return index
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to a dictionary for storage or display
        
        Returns:
            Dictionary of settings
        """
        return {
            'index_type': self.index_type,
            'metric_type': self.metric_type,
            'embedding_dim': self.embedding_dim,
            'index_factory_string': self.index_factory_string,
            'use_index_gpu': self.use_index_gpu,
            'n_centroids': self.n_centroids,
            'hnsw_m': self.hnsw_m,
            'hnsw_ef_construction': self.hnsw_ef_construction,
            'hnsw_ef_search': self.hnsw_ef_search,
            'pq_m': self.pq_m,
            'pq_bits': self.pq_bits
        }
    
    def save(self) -> None:
        """Save settings to the database or configuration"""
        try:
            current_app.config['FAISS_SETTINGS'] = self.to_dict()
            # TODO: Persist to database if needed
            logger.info("FAISS settings saved")
        except Exception as e:
            logger.error(f"Could not save FAISS settings: {str(e)}")


class EmbeddingSettings:
    """
    Settings for embedding configuration
    """
    # Available data types
    DTYPES = {
        'float32': {
            'name': 'Float32',
            'description': 'Standard precision (4 bytes per dimension)',
            'dtype': np.float32
        },
        'float16': {
            'name': 'Float16',
            'description': 'Half precision (2 bytes per dimension, saves memory)',
            'dtype': np.float16
        },
        'int8': {
            'name': 'Int8', 
            'description': 'Quantized to 8-bit integers (1 byte per dimension)',
            'dtype': np.int8
        }
    }
    
    def __init__(self):
        """Initialize embedding settings with defaults"""
        # Default settings
        self.embedding_dim = 384  # Will be updated based on active models
        self.dtype = 'float32'
        self.normalize_vectors = True  # Whether to normalize vectors before storage
        
        # Load settings from database or file if available
        self._load_settings()
    
    def _load_settings(self) -> None:
        """Load settings from the database or config file"""
        try:
            # Try to load from application config
            if hasattr(current_app, 'config') and 'EMBEDDING_SETTINGS' in current_app.config:
                settings = current_app.config['EMBEDDING_SETTINGS']
                self._update_from_dict(settings)
            else:
                # Try to find the dimensions from the default embedding model
                model = EmbeddingModel.query.filter_by(is_default=True).first()
                if model:
                    self.embedding_dim = model.dimension
        except Exception as e:
            logger.warning(f"Could not load embedding settings: {str(e)}")
    
    def _update_from_dict(self, settings: Dict[str, Any]) -> None:
        """Update settings from a dictionary"""
        for key, value in settings.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_numpy_dtype(self) -> np.dtype:
        """
        Get the numpy dtype for embeddings
        
        Returns:
            Numpy dtype
        """
        return self.DTYPES[self.dtype]['dtype']
    
    def process_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Process a vector according to settings (normalize, convert dtype)
        
        Args:
            vector: Input vector
            
        Returns:
            Processed vector
        """
        # Convert dtype if needed
        if vector.dtype != self.get_numpy_dtype():
            vector = vector.astype(self.get_numpy_dtype())
        
        # Normalize if needed
        if self.normalize_vectors:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        
        return vector
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to a dictionary for storage or display
        
        Returns:
            Dictionary of settings
        """
        return {
            'embedding_dim': self.embedding_dim,
            'dtype': self.dtype,
            'normalize_vectors': self.normalize_vectors
        }
    
    def save(self) -> None:
        """Save settings to the database or configuration"""
        try:
            current_app.config['EMBEDDING_SETTINGS'] = self.to_dict()
            # TODO: Persist to database if needed
            logger.info("Embedding settings saved")
        except Exception as e:
            logger.error(f"Could not save embedding settings: {str(e)}")


class AdminSettings:
    """
    Main admin settings class for managing system-wide settings
    """
    def __init__(self):
        """Initialize all settings components"""
        self.faiss_settings = FaissSettings()
        self.embedding_settings = EmbeddingSettings()
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Convert all settings to a dictionary
        
        Returns:
            Dictionary of all settings
        """
        return {
            'faiss': self.faiss_settings.to_dict(),
            'embedding': self.embedding_settings.to_dict()
        }
    
    def save(self) -> None:
        """Save all settings"""
        self.faiss_settings.save()
        self.embedding_settings.save()
    
    def rebuild_index(self, vector_store: Any = None) -> bool:
        """
        Rebuild the FAISS index with current settings
        
        Args:
            vector_store: Optional VectorStore instance
            
        Returns:
            True if successful
        """
        from vector_store import VectorStore
        from routes import get_vector_store
        
        try:
            # Get vector store if not provided
            vs = vector_store or get_vector_store()
            
            # Rebuild all indices
            for model_id in vs.indices:
                vs._rebuild_index(model_id)
            
            logger.info("All indices rebuilt successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to rebuild indices: {str(e)}")
            return False