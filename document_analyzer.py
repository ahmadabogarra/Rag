
import re
import logging
import json
from typing import Dict, Any, List
from app import db
from models import Document, Chunk, Metadata
from document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    def __init__(self, document_processor: DocumentProcessor):
        self.document_processor = document_processor

    def analyze_document(self, document_id: str) -> Dict[str, Any]:
        document = Document.query.get_or_404(document_id)
        metadata = {m.key: m.value for m in Metadata.query.filter_by(document_id=document_id).all()}
        
        # Get document content
        content = document.content
        structure_type = metadata.get('detected_structure', 'unstructured')
        
        fields = []
        if structure_type == 'semi_structured':
            try:
                # Parse JSON content
                json_data = json.loads(content)
                # Extract field names from first item if it's an array
                if isinstance(json_data, list) and len(json_data) > 0:
                    json_data = json_data[0]
                fields = self._extract_json_fields(json_data)
            except:
                fields = []
        
        return {
            'document': document,
            'metadata': metadata,
            'fields': fields,
            'structure_type': structure_type
        }

    def _extract_json_fields(self, data: Any, parent_path: str = '') -> List[Dict[str, str]]:
        """Extract field names and types from JSON data"""
        fields = []
        if isinstance(data, dict):
            for key, value in data.items():
                full_path = f"{parent_path}.{key}" if parent_path else key
                field_type = type(value).__name__
                fields.append({
                    'name': full_path,
                    'type': field_type
                })
                
                if isinstance(value, (dict, list)):
                    fields.extend(self._extract_json_fields(value, full_path))
        elif isinstance(data, list) and data:
            fields.extend(self._extract_json_fields(data[0], parent_path))
            
        return fields

    def save_processing_config(self, document_id: str, config: Dict[str, Any]) -> bool:
        try:
            document = Document.query.get_or_404(document_id)
            for key, value in config.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                Metadata.query.filter_by(
                    document_id=document_id,
                    key=key
                ).delete()
                db.session.add(Metadata(
                    document_id=document_id,
                    key=key,
                    value=str(value)
                ))
            db.session.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving config: {e}")
            db.session.rollback()
            return False

    def process_document(self, document_id: str, config: Dict[str, Any]) -> bool:
        try:
            # Save config first
            if not self.save_processing_config(document_id, config):
                return False
                
            # Process document with new config
            document = Document.query.get_or_404(document_id)
            self.document_processor.update_document(
                document_id=document_id,
                name=document.name,
                content=document.content,
                metadata=config
            )
            return True
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return False

    def generate_preview(self, document_id: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate preview chunks based on configuration"""
        document = Document.query.get_or_404(document_id)
        chunks = []
        
        try:
            # Use document processor to generate chunks
            chunks = self.document_processor.generate_chunks(
                content=document.content,
                structure_type=config['detected_structure'],
                chunking_config=config['chunking_config']
            )
            
            # Convert chunks to preview format
            preview_chunks = []
            for i, chunk in enumerate(chunks):
                preview_chunks.append({
                    'index': i,
                    'content': chunk['content'],
                    'metadata': chunk.get('metadata', {})
                })
            
            return preview_chunks
            
        except Exception as e:
            logger.error(f"Error generating preview: {e}")
            return []
