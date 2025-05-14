
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
        
        # Extract custom metadata from existing metadata
        custom_metadata = {}
        for key, value in metadata.items():
            if not key.startswith('detected_') and key not in ['store_full_content', 'chunking_config', 'embedding_config']:
                custom_metadata[key] = value

        # Initialize chunking configuration with defaults
        chunking_config = {
            'structured': {
                'method': 'row_based',
                'min_rows_per_chunk': 1,
                'max_rows_per_chunk': 10
            },
            'semi_structured': {
                'method': 'semantic_elements',
                'element_path': '',
                'preserve_hierarchy': True
            },
            'unstructured': {
                'method': 'paragraph_based',
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'regex_pattern': r'\n\s*\n|\r\n\s*\r\n'
            }
        }
        
        # Try to get existing chunking config from metadata
        try:
            if 'chunking_config' in metadata:
                saved_config = json.loads(metadata['chunking_config'])
                # Update only the relevant structure type config
                if structure_type in saved_config:
                    chunking_config[structure_type].update(saved_config[structure_type])
        except:
            pass

        return {
            'document': document,
            'metadata': metadata,
            'fields': fields,
            'structure_type': structure_type,
            'custom_metadata': custom_metadata,
            'store_full_content': metadata.get('store_full_content', 'false') == 'true',
            'detected_structure_name': metadata.get('detected_structure', 'unstructured'),
            'detection_confidence': float(metadata.get('detection_confidence', '0.0')),
            'detected_language': metadata.get('detected_language', 'ar'),
            'chunking_config': chunking_config
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
