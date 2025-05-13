import uuid
import datetime
from app import db
from sqlalchemy.dialects.sqlite import JSON

class Document(db.Model):
    """Document model to store uploaded documents and their metadata"""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    mime_type = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    chunks = db.relationship('Chunk', backref='document', cascade='all, delete-orphan')
    doc_metadata = db.relationship('Metadata', backref='document', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Document {self.name}>'

class Chunk(db.Model):
    """Chunk model to store document chunks for vector embeddings"""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = db.Column(db.String(36), db.ForeignKey('document.id'), nullable=False)
    content = db.Column(db.Text, nullable=False)
    chunk_index = db.Column(db.Integer, nullable=False)
    start_pos = db.Column(db.Integer, nullable=False)
    end_pos = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    embeddings = db.relationship('ChunkEmbedding', backref='chunk', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Chunk {self.id} - Document {self.document_id}>'

class Metadata(db.Model):
    """Metadata model to store document metadata"""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = db.Column(db.String(36), db.ForeignKey('document.id'), nullable=False)
    key = db.Column(db.String(100), nullable=False)
    value = db.Column(db.String(255), nullable=False)
    
    def __repr__(self):
        return f'<Metadata {self.key}={self.value} - Document {self.document_id}>'

class EmbeddingModel(db.Model):
    """Embedding model configuration"""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = db.Column(db.String(100), nullable=False, unique=True)
    model_type = db.Column(db.String(50), nullable=False)  # sentence-transformers, openai, etc.
    model_path = db.Column(db.String(255), nullable=False)  # Path or identifier for the model
    dimension = db.Column(db.Integer, nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    is_default = db.Column(db.Boolean, default=False)
    config = db.Column(JSON, nullable=True)  # Additional configuration
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    embeddings = db.relationship('ChunkEmbedding', backref='model', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<EmbeddingModel {self.name}>'

class ChunkEmbedding(db.Model):
    """Store vector embeddings for document chunks"""
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    chunk_id = db.Column(db.String(36), db.ForeignKey('chunk.id'), nullable=False)
    model_id = db.Column(db.String(36), db.ForeignKey('embedding_model.id'), nullable=False)
    embedding_data = db.Column(db.LargeBinary, nullable=False)  # Serialized vector data
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    __table_args__ = (
        db.UniqueConstraint('chunk_id', 'model_id', name='_chunk_model_uc'),
    )
    
    def __repr__(self):
        return f'<ChunkEmbedding {self.id} - Chunk {self.chunk_id}>'
