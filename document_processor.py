import re
import logging
import os
from typing import List, Dict, Tuple, Optional, Any
from app import db
from models import Document, Chunk, Metadata
from schema_detector import DocumentSchemaDetector

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles document processing, chunking, and metadata extraction
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor with chunking parameters
        
        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.schema_detector = DocumentSchemaDetector()
    
    def process_document(self, 
                        name: str, 
                        content: str, 
                        mime_type: str, 
                        metadata: Optional[Dict[str, str]] = None,
                        override_type: Optional[str] = None) -> Document:
        """
        Process a document - save to database, extract metadata, and create chunks
        
        Args:
            name: Document name
            content: Document content/text
            mime_type: MIME type of the document
            metadata: Optional dictionary of metadata key-value pairs
            override_type: Optional manual override for document structure type
            
        Returns:
            Document object
        """
        # Detect document structure
        file_extension = os.path.splitext(name)[1].lower().lstrip('.') if '.' in name else ''
        schema_info = self.schema_detector.detect_schema(
            raw_text=content, 
            mime_type=mime_type,
            override_type=override_type
        )
        
        detected_type = schema_info["detected_type"]
        confidence = schema_info["confidence_score"]
        
        # Create new document
        document = Document(
            name=name,
            content=content,
            mime_type=mime_type
        )
        
        # Save document to get an ID
        db.session.add(document)
        db.session.commit()
        
        # Add metadata if provided
        if metadata is None:
            metadata = {}
            
        # Add schema detection metadata
        metadata.update({
            "detected_structure": detected_type,
            "detection_confidence": str(confidence)
        })
        
        # Add structure hint if available
        if "structure_hint" in schema_info:
            metadata["structure_hint"] = schema_info["structure_hint"]
        
        self._add_metadata(document.id, metadata)
        
        # Create chunks for the document using detected structure type
        self._create_chunks(document.id, content, detected_type)
        
        logger.info(f"Processed document: {name} (ID: {document.id}) as {detected_type}")
        return document
    
    def update_document(self, 
                        document_id: str, 
                        name: Optional[str] = None, 
                        content: Optional[str] = None, 
                        metadata: Optional[Dict[str, str]] = None) -> Document:
        """
        Update an existing document, refresh chunks if content changed
        
        Args:
            document_id: ID of document to update
            name: New document name (optional)
            content: New document content (optional)
            metadata: New metadata key-value pairs (optional)
            
        Returns:
            Updated Document object
        """
        document = Document.query.get(document_id)
        if not document:
            raise ValueError(f"Document with ID {document_id} not found")
        
        content_changed = False
        
        # Update fields if provided
        if name:
            document.name = name
            
        if content and content != document.content:
            document.content = content
            content_changed = True
        
        # Update database
        db.session.commit()
        
        # Update metadata if provided
        if metadata:
            # Clear existing metadata
            Metadata.query.filter_by(document_id=document_id).delete()
            db.session.commit()
            
            # Add new metadata
            self._add_metadata(document_id, metadata)
        
        # If content changed, recreate chunks
        if content_changed:
            # Delete existing chunks
            Chunk.query.filter_by(document_id=document_id).delete()
            db.session.commit()
            
            # Create new chunks
            self._create_chunks(document_id, document.content)
            
            logger.info(f"Updated document content and chunks: {document.name} (ID: {document_id})")
        else:
            logger.info(f"Updated document metadata: {document.name} (ID: {document_id})")
            
        return document
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all associated chunks and metadata
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if successful
        """
        document = Document.query.get(document_id)
        if not document:
            raise ValueError(f"Document with ID {document_id} not found")
        
        # The cascade delete will handle chunks and metadata
        db.session.delete(document)
        db.session.commit()
        
        logger.info(f"Deleted document: {document.name} (ID: {document_id})")
        return True
    
    def _add_metadata(self, document_id: str, metadata: Dict[str, str]) -> None:
        """
        Add metadata key-value pairs to a document
        
        Args:
            document_id: Document ID
            metadata: Dictionary of metadata key-value pairs
        """
        for key, value in metadata.items():
            meta = Metadata(
                document_id=document_id,
                key=key,
                value=str(value)
            )
            db.session.add(meta)
        
        db.session.commit()
    
    def _create_chunks(self, document_id: str, content: str) -> List[Chunk]:
        """
        Split document content into chunks with specified size and overlap
        
        Args:
            document_id: Document ID
            content: Document content to chunk
            
        Returns:
            List of created Chunk objects
        """
        # Calculate positions for chunks
        chunks_data = self._split_text(content, self.chunk_size, self.chunk_overlap)
        
        created_chunks = []
        for i, (chunk_text, start_pos, end_pos) in enumerate(chunks_data):
            chunk = Chunk(
                document_id=document_id,
                content=chunk_text,
                chunk_index=i,
                start_pos=start_pos,
                end_pos=end_pos
            )
            db.session.add(chunk)
            created_chunks.append(chunk)
        
        db.session.commit()
        logger.info(f"Created {len(created_chunks)} chunks for document {document_id}")
        return created_chunks
    
    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int, int]]:
        """
        Split text into chunks with specified size and overlap
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of tuples (chunk_text, start_position, end_position)
        """
        # Use sentence-aware splitting to avoid cutting sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        start_pos = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size and we already have content,
            # finalize the current chunk
            if current_chunk and (current_length + sentence_length > chunk_size):
                chunk_text = " ".join(current_chunk)
                end_pos = start_pos + len(chunk_text)
                chunks.append((chunk_text, start_pos, end_pos))
                
                # Start a new chunk with overlap
                overlap_text = chunk_text[max(0, len(chunk_text) - chunk_overlap):]
                start_pos = end_pos - len(overlap_text)
                current_chunk = []
                current_length = 0
                
                # If there was overlap text, add it to the new chunk
                if overlap_text:
                    current_chunk.append(overlap_text)
                    current_length = len(overlap_text)
            
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for the space
        
        # Add the final chunk if there's anything left
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            end_pos = start_pos + len(chunk_text)
            chunks.append((chunk_text, start_pos, end_pos))
        
        return chunks
    
    def set_chunking_parameters(self, chunk_size: int, chunk_overlap: int) -> None:
        """
        Update chunking parameters
        
        Args:
            chunk_size: New chunk size
            chunk_overlap: New chunk overlap
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Updated chunking parameters: size={chunk_size}, overlap={chunk_overlap}")
