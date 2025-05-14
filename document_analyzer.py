import re
import os
import json
import logging
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from langdetect import detect
from app import db
from models import Document, Chunk, Metadata, EmbeddingModel
from document_processor import DocumentProcessor
from schema_detector import DocumentSchemaDetector

logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    """
    Analyzes documents to detect structure, language, and fields,
    and provides intelligent chunking strategies.
    """
    
    def __init__(self, document_processor: Optional[DocumentProcessor] = None):
        """
        Initialize the document analyzer.
        
        Args:
            document_processor: Optional DocumentProcessor to use
        """
        self.document_processor = document_processor or DocumentProcessor()
        self.schema_detector = DocumentSchemaDetector()
        
        # Default chunking configurations
        self.default_configs = {
            "structured": {
                "method": "row_based",
                "min_rows_per_chunk": 1,
                "max_rows_per_chunk": 10,
            },
            "semi_structured": {
                "method": "semantic_elements",
                "element_path": "",
                "preserve_hierarchy": True,
            },
            "unstructured": {
                "method": "paragraph_based",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "regex_pattern": r"\n\s*\n|\r\n\s*\r\n"
            }
        }
        
        # Default embedding configuration
        self.default_embedding_config = {
            "dtype": "float32",
            "normalize": True,
            "max_tokens_per_chunk": 512
        }
    
    def analyze_document(self, document_id: str) -> Dict[str, Any]:
        """
        Analyze a document to detect its structure, language, and fields.
        
        Args:
            document_id: ID of the document to analyze
            
        Returns:
            Analysis results including detected structure, language, fields, etc.
        """
        document = Document.query.get_or_404(document_id)
        content = document.content
        
        # Get existing metadata if any
        metadata_dict = {}
        for meta in Metadata.query.filter_by(document_id=document_id).all():
            metadata_dict[meta.key] = meta.value
        
        # Get existing processing configuration if available
        processing_config = metadata_dict.get('processing_config')
        if processing_config:
            try:
                processing_config = json.loads(processing_config)
            except:
                processing_config = None
        
        # Detect document structure if not already defined
        if not processing_config or 'detected_structure' not in processing_config:
            structure_info = self.schema_detector.detect_schema(raw_text=content, mime_type=document.mime_type)
            detected_structure = structure_info["detected_type"]
            confidence_score = structure_info["confidence_score"]
            structure_hint = structure_info.get("structure_hint", "")
        else:
            detected_structure = processing_config.get('detected_structure', 'unstructured')
            confidence_score = float(processing_config.get('detection_confidence', 0))
            structure_hint = processing_config.get('structure_hint', '')
        
        # Detect language if not already defined
        if not processing_config or 'detected_language' not in processing_config:
            detected_language = self._detect_language(content)
        else:
            detected_language = processing_config.get('detected_language', 'en')
        
        # Detect fields based on document structure
        if not processing_config or 'detected_fields' not in processing_config:
            detected_fields = self._detect_fields(content, detected_structure)
            embedding_fields = self._suggest_embedding_fields(detected_fields, detected_structure)
            metadata_fields = self._suggest_metadata_fields(detected_fields, detected_structure)
        else:
            detected_fields = processing_config.get('detected_fields', [])
            embedding_fields = processing_config.get('embedding_fields', [])
            metadata_fields = processing_config.get('metadata_fields', [])
        
        # Get chunking configuration
        chunking_config = processing_config.get('chunking_config', self.default_configs)
        
        # Get embedding configuration
        embedding_config = processing_config.get('embedding_config', self.default_embedding_config)
        
        # Get custom metadata
        custom_metadata = {}
        for key, value in metadata_dict.items():
            if key.startswith('custom_'):
                custom_metadata[key[7:]] = value
        
        # Get store_full_content setting
        store_full_content = processing_config.get('store_full_content', True)
        
        # Get appropriate models based on language
        available_models = self._get_available_models(detected_language)
        selected_model_id = processing_config.get('selected_model_id')
        
        if not selected_model_id and available_models:
            selected_model_id = available_models[0].id
            
        # Generate preview chunks if needed
        preview_chunks = []
        if processing_config and 'preview_chunks' in processing_config:
            preview_chunks = processing_config['preview_chunks']
        else:
            # Generate a few preview chunks for the UI
            preview_chunks = self._generate_preview_chunks(
                content, 
                detected_structure, 
                chunking_config[detected_structure]
            )
        
        # Put everything together
        result = {
            "document": document,
            "detected_structure": detected_structure,
            "detected_structure_name": self._get_structure_name(detected_structure),
            "detection_confidence": confidence_score,
            "structure_hint": structure_hint,
            "detected_language": detected_language,
            "detected_fields": detected_fields,
            "embedding_fields": embedding_fields,
            "metadata_fields": metadata_fields,
            "chunking_config": chunking_config,
            "embedding_config": embedding_config,
            "custom_metadata": custom_metadata,
            "store_full_content": store_full_content,
            "available_models": available_models,
            "selected_model_id": selected_model_id,
            "preview_chunks": preview_chunks,
            "chunks": Chunk.query.filter_by(document_id=document_id).all()
        }
        
        return result
    
    def save_processing_config(self, document_id: str, config: Dict[str, Any]) -> bool:
        """
        Save document processing configuration as metadata.
        
        Args:
            document_id: ID of the document
            config: Processing configuration
            
        Returns:
            True if successful
        """
        try:
            # Convert to JSON string
            config_json = json.dumps(config)
            
            # Check if config metadata already exists
            existing_meta = Metadata.query.filter_by(
                document_id=document_id, key='processing_config').first()
            
            if existing_meta:
                existing_meta.value = config_json
            else:
                # Create new metadata entry
                meta = Metadata(
                    document_id=document_id,
                    key='processing_config',
                    value=config_json
                )
                db.session.add(meta)
            
            # Save custom metadata
            if 'custom_metadata' in config:
                for key, value in config['custom_metadata'].items():
                    custom_key = f'custom_{key}'
                    existing_custom = Metadata.query.filter_by(
                        document_id=document_id, key=custom_key).first()
                    
                    if existing_custom:
                        existing_custom.value = value
                    else:
                        meta = Metadata(
                            document_id=document_id,
                            key=custom_key,
                            value=value
                        )
                        db.session.add(meta)
            
            db.session.commit()
            return True
        except Exception as e:
            logger.error(f"Error saving processing config: {str(e)}")
            db.session.rollback()
            return False
    
    def process_document(self, document_id: str, config: Dict[str, Any]) -> bool:
        """
        Process a document according to the specified configuration.
        
        Args:
            document_id: ID of the document
            config: Processing configuration
            
        Returns:
            True if successful
        """
        try:
            # Save the configuration first
            self.save_processing_config(document_id, config)
            
            # Get document
            document = Document.query.get_or_404(document_id)
            
            # Extract key configuration values
            structure_type = config.get('detected_structure', 'unstructured')
            chunking_config = config.get('chunking_config', {}).get(structure_type, {})
            embedding_fields = config.get('embedding_fields', [])
            metadata_fields = config.get('metadata_fields', [])
            store_full_content = config.get('store_full_content', True)
            
            # Prepare metadata to keep
            metadata = {}
            
            # Add structure information to metadata
            metadata.update({
                'detected_structure': structure_type,
                'detection_confidence': str(config.get('detection_confidence', 100)),
                'structure_hint': config.get('structure_hint', '')
            })
            
            # Add language information
            metadata['detected_language'] = config.get('detected_language', 'en')
            
            # Add custom metadata if present
            if 'custom_metadata' in config:
                for key, value in config['custom_metadata'].items():
                    metadata[key] = value
            
            # Generate content for processing based on selected embedding fields
            if embedding_fields and structure_type != 'unstructured':
                # For structured/semi-structured, combine selected fields
                try:
                    processed_content = self._extract_and_combine_fields(
                        document.content, structure_type, embedding_fields)
                except:
                    # Fallback to full content
                    processed_content = document.content
                    logger.warning(f"Could not extract fields from document {document_id}, using full content")
            else:
                # For unstructured, use the full content
                processed_content = document.content
            
            # Store selected metadata fields if store_full_content is not enabled
            if not store_full_content and metadata_fields and structure_type != 'unstructured':
                try:
                    extracted_metadata = self._extract_metadata_fields(
                        document.content, structure_type, metadata_fields)
                    metadata.update(extracted_metadata)
                except:
                    logger.warning(f"Could not extract metadata fields from document {document_id}")
            
            # If store_full_content is enabled, add the original content to metadata
            if store_full_content:
                metadata['full_content'] = 'true'
            
            # Update document with the processed content and metadata
            # This will trigger chunking based on the document structure
            self.document_processor.update_document(
                document_id=document_id,
                content=processed_content,
                metadata=metadata,
                structure_type=structure_type,
                chunking_config=chunking_config
            )
            
            # Process completed successfully
            return True
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return False
    
    def generate_preview(self, document_id: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate preview chunks based on configuration without saving.
        
        Args:
            document_id: ID of the document
            config: Processing configuration
            
        Returns:
            List of preview chunks
        """
        try:
            document = Document.query.get_or_404(document_id)
            structure_type = config.get('detected_structure', 'unstructured')
            chunking_config = config.get('chunking_config', {}).get(structure_type, {})
            
            # Generate preview chunks
            preview_chunks = self._generate_preview_chunks(
                document.content, structure_type, chunking_config, max_chunks=5)
                
            return preview_chunks
        except Exception as e:
            logger.error(f"Error generating preview: {str(e)}")
            return []
    
    def _detect_language(self, content: str) -> str:
        """
        Detect the language of content.
        
        Args:
            content: Text content
            
        Returns:
            Language code (e.g., 'en', 'ar', etc.)
        """
        try:
            # Sample the content to speed up detection
            sample = content[:min(5000, len(content))]
            return detect(sample)
        except:
            # Default to English if detection fails
            return 'en'
    
    def _detect_fields(self, content: str, structure_type: str) -> List[Dict[str, str]]:
        """
        Detect fields in structured or semi-structured content.
        
        Args:
            content: Document content
            structure_type: Document structure type
            
        Returns:
            List of detected fields with name and description
        """
        fields = []
        
        if structure_type == 'unstructured':
            # For unstructured text, just one field for the content
            fields.append({
                'name': 'content',
                'description': 'Full text content',
                'sample': content[:50] + '...' if len(content) > 50 else content
            })
            return fields
        
        # For structured data, try to extract column headers
        if structure_type == 'structured':
            try:
                import csv
                from io import StringIO
                
                # Try to detect CSV format
                first_lines = content.split('\n', 10)
                header_line = first_lines[0] if first_lines else ''
                
                # Detect dialect
                try:
                    dialect = csv.Sniffer().sniff(header_line)
                    has_header = csv.Sniffer().has_header(content[:1000])
                except:
                    dialect = csv.excel
                    has_header = True
                
                # Parse header
                if has_header:
                    reader = csv.reader(StringIO(header_line), dialect)
                    headers = next(reader, [])
                    
                    # Get first data row for samples
                    data_row = []
                    if len(first_lines) > 1:
                        reader = csv.reader(StringIO(first_lines[1]), dialect)
                        data_row = next(reader, [])
                    
                    # Create fields
                    for i, header in enumerate(headers):
                        sample = data_row[i] if i < len(data_row) else ''
                        fields.append({
                            'name': header,
                            'description': f'Column {i+1}',
                            'sample': sample
                        })
                
                if not fields:
                    # If header detection failed, create generic field
                    fields.append({
                        'name': 'content',
                        'description': 'CSV content',
                        'sample': header_line
                    })
            except Exception as e:
                logger.warning(f"Error detecting CSV fields: {str(e)}")
                fields.append({
                    'name': 'content',
                    'description': 'CSV content',
                    'sample': content[:50] + '...' if len(content) > 50 else content
                })
                
        # For semi-structured data, try to extract fields
        elif structure_type == 'semi_structured':
            # Check if it's JSON
            if content.strip().startswith('{') or content.strip().startswith('['):
                try:
                    data = json.loads(content)
                    
                    if isinstance(data, dict):
                        # Single JSON object
                        for key, value in data.items():
                            sample = str(value)
                            if isinstance(value, (dict, list)):
                                sample = json.dumps(value)[:50] + '...'
                            elif isinstance(value, str) and len(sample) > 50:
                                sample = sample[:50] + '...'
                                
                            fields.append({
                                'name': key,
                                'description': f'JSON field: {key}',
                                'sample': sample
                            })
                    elif isinstance(data, list) and data and isinstance(data[0], dict):
                        # Array of objects - use first object for fields
                        for key, value in data[0].items():
                            sample = str(value)
                            if isinstance(value, (dict, list)):
                                sample = json.dumps(value)[:50] + '...'
                            elif isinstance(value, str) and len(sample) > 50:
                                sample = sample[:50] + '...'
                                
                            fields.append({
                                'name': key,
                                'description': f'JSON array field: {key}',
                                'sample': sample
                            })
                except Exception as e:
                    logger.warning(f"Error parsing JSON: {str(e)}")
                    fields.append({
                        'name': 'content',
                        'description': 'JSON content',
                        'sample': content[:50] + '...' if len(content) > 50 else content
                    })
            
            # Check if it's XML-like
            elif content.strip().startswith('<') and '>' in content:
                try:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(content)
                    
                    # Get first-level elements
                    for child in list(root)[:10]:  # Limit to first 10 elements
                        tag = child.tag
                        if tag.startswith('{'):  # Remove namespace
                            tag = tag.split('}', 1)[1]
                            
                        # Get text content or attributes
                        sample = child.text or ''
                        if not sample and child.attrib:
                            sample = str(child.attrib)
                        
                        if len(sample) > 50:
                            sample = sample[:50] + '...'
                            
                        fields.append({
                            'name': tag,
                            'description': f'XML element: {tag}',
                            'sample': sample
                        })
                except Exception as e:
                    logger.warning(f"Error parsing XML: {str(e)}")
                    fields.append({
                        'name': 'content',
                        'description': 'XML content',
                        'sample': content[:50] + '...' if len(content) > 50 else content
                    })
            
            # Check if it's Markdown
            elif re.search(r'^#+ ', content, re.MULTILINE):
                # Extract headers as fields
                headers = re.findall(r'^(#+) (.+)$', content, re.MULTILINE)
                
                if headers:
                    for i, (level, header) in enumerate(headers[:10]):  # Limit to first 10 headers
                        level_num = len(level)
                        fields.append({
                            'name': f'header_level_{level_num}',
                            'description': f'Level {level_num} header: {header}',
                            'sample': header
                        })
                    
                    # Add content field
                    fields.append({
                        'name': 'content',
                        'description': 'Markdown content',
                        'sample': content[:50] + '...' if len(content) > 50 else content
                    })
                else:
                    fields.append({
                        'name': 'content',
                        'description': 'Markdown content',
                        'sample': content[:50] + '...' if len(content) > 50 else content
                    })
            
            # Fallback for other semi-structured formats
            if not fields:
                fields.append({
                    'name': 'content',
                    'description': 'Semi-structured content',
                    'sample': content[:50] + '...' if len(content) > 50 else content
                })
        
        return fields
    
    def _suggest_embedding_fields(self, fields: List[Dict[str, str]], structure_type: str) -> List[str]:
        """
        Suggest fields to use for embedding based on content.
        
        Args:
            fields: List of detected fields
            structure_type: Document structure type
            
        Returns:
            List of field names suggested for embedding
        """
        # For unstructured, always use content
        if structure_type == 'unstructured':
            return ['content']
        
        # For structured/semi-structured, look for text fields likely containing meaningful content
        meaningful_fields = []
        
        # Keywords that might indicate important text fields
        important_keywords = [
            'title', 'name', 'description', 'summary', 'content', 'text',
            'body', 'abstract', 'detail', 'headline', 'subject', 'message',
            'comment', 'note', 'info', 'article'
        ]
        
        # First, check for explicitly important fields
        for field in fields:
            field_name = field['name'].lower()
            if any(keyword in field_name for keyword in important_keywords):
                meaningful_fields.append(field['name'])
        
        # If none found, use text fields with longer samples
        if not meaningful_fields:
            for field in fields:
                sample = field.get('sample', '')
                if isinstance(sample, str) and len(sample) > 30:  # Arbitrary threshold for meaningful text
                    meaningful_fields.append(field['name'])
        
        # Fallback to first field if still no meaningful fields found
        if not meaningful_fields and fields:
            meaningful_fields.append(fields[0]['name'])
        
        return meaningful_fields
    
    def _suggest_metadata_fields(self, fields: List[Dict[str, str]], structure_type: str) -> List[str]:
        """
        Suggest fields to store as metadata based on content.
        
        Args:
            fields: List of detected fields
            structure_type: Document structure type
            
        Returns:
            List of field names suggested for metadata
        """
        # For unstructured, no extra metadata fields
        if structure_type == 'unstructured':
            return []
        
        # For structured/semi-structured, suggest fields that might be useful for filtering
        metadata_fields = []
        
        # Keywords that might indicate useful metadata fields
        metadata_keywords = [
            'id', 'date', 'time', 'category', 'type', 'status', 'tag', 'label',
            'author', 'source', 'reference', 'version', 'language', 'region',
            'price', 'cost', 'amount', 'quantity', 'count', 'size', 'color'
        ]
        
        # Check for explicitly useful metadata fields
        for field in fields:
            field_name = field['name'].lower()
            if any(keyword in field_name for keyword in metadata_keywords):
                metadata_fields.append(field['name'])
        
        # Add ID-like short fields
        for field in fields:
            field_name = field['name'].lower()
            sample = field.get('sample', '')
            if field_name not in metadata_fields and isinstance(sample, str) and len(sample) < 30:
                metadata_fields.append(field['name'])
        
        return metadata_fields
    
    def _get_structure_name(self, structure_type: str) -> str:
        """
        Get a human-readable name for a structure type.
        
        Args:
            structure_type: Structure type ('structured', 'semi_structured', 'unstructured')
            
        Returns:
            Human-readable name
        """
        names = {
            'structured': 'ثابت البنية',
            'semi_structured': 'شبه منظم',
            'unstructured': 'نص حر'
        }
        return names.get(structure_type, structure_type)
    
    def _get_available_models(self, language_code: str) -> List[EmbeddingModel]:
        """
        Get available embedding models suitable for the detected language.
        
        Args:
            language_code: Detected language code
            
        Returns:
            List of suitable embedding models
        """
        # Get all active models
        all_models = EmbeddingModel.query.filter_by(is_active=True).all()
        
        # If no models, return empty list
        if not all_models:
            return []
        
        # Categorize models by language support
        language_specific_models = []
        multilingual_models = []
        other_models = []
        
        # Languages that have specific models
        language_specific_codes = ['en', 'ar', 'zh', 'fr', 'de', 'es', 'ru', 'ja']
        
        for model in all_models:
            model_name = model.name.lower()
            model_path = model.model_path.lower()
            
            # Check for multilingual models
            if 'multilingual' in model_name or 'multilingual' in model_path:
                multilingual_models.append(model)
            # Check for language-specific models
            elif language_code in language_specific_codes and language_code in model_name:
                language_specific_models.append(model)
            # Other models
            else:
                other_models.append(model)
        
        # Prioritize language-specific models, then multilingual, then others
        if language_specific_models:
            return language_specific_models + multilingual_models + other_models
        else:
            return multilingual_models + other_models
    
    def _extract_and_combine_fields(self, content: str, structure_type: str, 
                                   field_names: List[str]) -> str:
        """
        Extract and combine specified fields from structured/semi-structured content.
        
        Args:
            content: Document content
            structure_type: Document structure type
            field_names: Names of fields to extract
            
        Returns:
            Combined content from specified fields
        """
        if not field_names:
            return content
            
        if structure_type == 'structured':
            try:
                import csv
                from io import StringIO
                
                # Try to parse as CSV
                reader = csv.DictReader(StringIO(content))
                rows = list(reader)
                
                # Combine specified fields from each row
                combined_content = []
                for row in rows:
                    row_content = []
                    for field in field_names:
                        if field in row:
                            row_content.append(f"{field}: {row[field]}")
                    combined_content.append(" | ".join(row_content))
                
                return "\n".join(combined_content)
            except:
                # Fallback to original content
                return content
                
        elif structure_type == 'semi_structured':
            if content.strip().startswith('{') or content.strip().startswith('['):
                try:
                    data = json.loads(content)
                    
                    if isinstance(data, dict):
                        # Single JSON object
                        combined_content = []
                        for field in field_names:
                            if field in data:
                                value = data[field]
                                if isinstance(value, (dict, list)):
                                    value = json.dumps(value)
                                combined_content.append(f"{field}: {value}")
                        return "\n".join(combined_content)
                        
                    elif isinstance(data, list) and data:
                        # Array of objects
                        combined_content = []
                        for item in data:
                            if isinstance(item, dict):
                                item_content = []
                                for field in field_names:
                                    if field in item:
                                        value = item[field]
                                        if isinstance(value, (dict, list)):
                                            value = json.dumps(value)
                                        item_content.append(f"{field}: {value}")
                                if item_content:
                                    combined_content.append(" | ".join(item_content))
                        return "\n".join(combined_content)
                except:
                    # Fallback to original content
                    return content
            
            # Fallback for other formats
            return content
        
        # Fallback for any other case
        return content
    
    def _extract_metadata_fields(self, content: str, structure_type: str, 
                               field_names: List[str]) -> Dict[str, str]:
        """
        Extract specified fields as metadata from structured/semi-structured content.
        
        Args:
            content: Document content
            structure_type: Document structure type
            field_names: Names of fields to extract as metadata
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}
        
        if not field_names:
            return metadata
            
        if structure_type == 'structured':
            try:
                import csv
                from io import StringIO
                
                # Try to parse as CSV
                reader = csv.DictReader(StringIO(content))
                rows = list(reader)
                
                # For CSV, store first row's metadata
                if rows:
                    for field in field_names:
                        if field in rows[0]:
                            metadata[f'meta_{field}'] = rows[0][field]
            except:
                pass
                
        elif structure_type == 'semi_structured':
            if content.strip().startswith('{') or content.strip().startswith('['):
                try:
                    data = json.loads(content)
                    
                    if isinstance(data, dict):
                        # Single JSON object
                        for field in field_names:
                            if field in data:
                                value = data[field]
                                if isinstance(value, (dict, list)):
                                    value = json.dumps(value)
                                elif not isinstance(value, str):
                                    value = str(value)
                                metadata[f'meta_{field}'] = value
                        
                    elif isinstance(data, list) and data and isinstance(data[0], dict):
                        # For array, use first item's metadata
                        for field in field_names:
                            if field in data[0]:
                                value = data[0][field]
                                if isinstance(value, (dict, list)):
                                    value = json.dumps(value)
                                elif not isinstance(value, str):
                                    value = str(value)
                                metadata[f'meta_{field}'] = value
                except:
                    pass
        
        return metadata
    
    def _generate_preview_chunks(self, content: str, structure_type: str, chunking_config: Dict[str, Any],
                               max_chunks: int = 3) -> List[Dict[str, Any]]:
        """
        Generate preview chunks based on content and chunking strategy.
        
        Args:
            content: Document content
            structure_type: Document structure type
            chunking_config: Chunking configuration
            max_chunks: Maximum number of preview chunks to generate
            
        Returns:
            List of preview chunks with content and metadata
        """
        preview_chunks = []
        
        # Get chunking method based on structure type
        if structure_type == 'structured':
            chunking_method = chunking_config.get('method', 'row_based')
            if chunking_method == 'row_based':
                chunks = self._preview_structured_row_based(
                    content, 
                    min_rows=chunking_config.get('min_rows_per_chunk', 1),
                    max_rows=chunking_config.get('max_rows_per_chunk', 10)
                )
            else:
                # Default to row-based
                chunks = self._preview_structured_row_based(content)
                
        elif structure_type == 'semi_structured':
            chunking_method = chunking_config.get('method', 'semantic_elements')
            if chunking_method == 'semantic_elements':
                chunks = self._preview_semi_structured_elements(content)
            elif chunking_method == 'custom_path':
                element_path = chunking_config.get('element_path', '')
                chunks = self._preview_semi_structured_custom_path(content, element_path)
            else:
                # Default to semantic elements
                chunks = self._preview_semi_structured_elements(content)
                
        else:  # unstructured
            chunking_method = chunking_config.get('method', 'fixed_size')
            if chunking_method == 'fixed_size':
                chunks = self._preview_unstructured_fixed_size(
                    content,
                    chunk_size=chunking_config.get('chunk_size', 1000),
                    chunk_overlap=chunking_config.get('chunk_overlap', 200)
                )
            elif chunking_method == 'sentence_based':
                chunks = self._preview_unstructured_sentence_based(content)
            elif chunking_method == 'paragraph_based':
                chunks = self._preview_unstructured_paragraph_based(content)
            elif chunking_method == 'regex_based':
                regex_pattern = chunking_config.get('regex_pattern', r'\n\s*\n|\r\n\s*\r\n')
                chunks = self._preview_unstructured_regex_based(content, regex_pattern)
            else:
                # Default to fixed size
                chunks = self._preview_unstructured_fixed_size(content)
        
        # Convert chunks to preview format
        for i, chunk_text in enumerate(chunks[:max_chunks]):
            preview_chunks.append({
                'index': i,
                'content': chunk_text,
                'length': len(chunk_text)
            })
            
        return preview_chunks
    
    def _preview_structured_row_based(self, content: str, min_rows: int = 1, max_rows: int = 10) -> List[str]:
        """Generate preview chunks for structured data using row-based chunking"""
        chunks = []
        try:
            import csv
            from io import StringIO
            
            # Try to parse as CSV
            first_line = content.split('\n', 1)[0]
            try:
                dialect = csv.Sniffer().sniff(first_line)
                has_header = csv.Sniffer().has_header(content[:1000])
            except:
                dialect = csv.excel
                has_header = True
            
            lines = content.split('\n')
            header = lines[0] if has_header else None
            data_start = 1 if has_header else 0
            
            # Process rows in chunks
            current_chunk = [header] if header else []
            row_count = 0
            
            for i in range(data_start, len(lines)):
                if not lines[i].strip():
                    continue
                    
                current_chunk.append(lines[i])
                row_count += 1
                
                # Check if we've reached max_rows
                if row_count >= max_rows:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [header] if header else []
                    row_count = 0
            
            # Add any remaining rows
            if row_count >= min_rows:
                chunks.append('\n'.join(current_chunk))
        except Exception as e:
            # Fallback: split by lines
            logger.warning(f"Error in structured chunking preview: {str(e)}")
            lines = content.split('\n')
            if len(lines) > 1:
                chunks = ['\n'.join(lines[:min(max_rows, len(lines))])]
            else:
                chunks = [content]
        
        return chunks
    
    def _preview_semi_structured_elements(self, content: str) -> List[str]:
        """Generate preview chunks for semi-structured data based on semantic elements"""
        chunks = []
        
        # Check if it's JSON
        if content.strip().startswith('{') or content.strip().startswith('['):
            try:
                data = json.loads(content)
                
                if isinstance(data, dict):
                    # For a single object, return as one chunk
                    chunks.append(json.dumps(data, indent=2))
                elif isinstance(data, list) and data:
                    # For array, return first few items
                    for item in data[:min(5, len(data))]:
                        chunks.append(json.dumps(item, indent=2))
            except:
                # Fallback: return first 1000 chars
                chunks.append(content[:min(1000, len(content))])
        
        # Check if it's XML-like
        elif content.strip().startswith('<') and '>' in content:
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(content)
                
                # Get first few child elements
                for child in list(root)[:5]:
                    chunks.append(ET.tostring(child, encoding='unicode'))
            except:
                # Fallback: return first 1000 chars
                chunks.append(content[:min(1000, len(content))])
        
        # Check if it's Markdown
        elif re.search(r'^#+ ', content, re.MULTILINE):
            # Split by headers
            sections = re.split(r'^(#+ .+)$', content, flags=re.MULTILINE)
            
            # Pair headers with their content
            for i in range(1, len(sections), 2):
                if i + 1 < len(sections):
                    chunks.append(sections[i] + '\n' + sections[i+1])
                else:
                    chunks.append(sections[i])
                    
            # If no chunks were created, fallback
            if not chunks:
                chunks.append(content[:min(1000, len(content))])
        
        # Fallback
        if not chunks:
            chunks.append(content[:min(1000, len(content))])
        
        return chunks
    
    def _preview_semi_structured_custom_path(self, content: str, element_path: str) -> List[str]:
        """Generate preview chunks for semi-structured data based on custom path"""
        chunks = []
        
        if not element_path:
            return self._preview_semi_structured_elements(content)
        
        # Check if it's JSON
        if content.strip().startswith('{') or content.strip().startswith('['):
            try:
                data = json.loads(content)
                
                # Simple path implementation for JSONPath-like syntax
                if element_path.startswith('$.'):
                    # Remove the leading $
                    path = element_path[2:].split('.')
                    
                    # Navigate to the specified path
                    current = data
                    for p in path:
                        # Handle array index notation like items[*]
                        if '[*]' in p:
                            p = p.split('[')[0]
                            if p in current and isinstance(current[p], list):
                                # Get items from array
                                for item in current[p][:5]:  # Limit to 5 items
                                    chunks.append(json.dumps(item, indent=2))
                                break
                        else:
                            if isinstance(current, dict) and p in current:
                                current = current[p]
                            else:
                                break
                    
                    # If we've reached the end of the path and haven't created chunks
                    if not chunks:
                        if isinstance(current, list):
                            for item in current[:5]:
                                chunks.append(json.dumps(item, indent=2))
                        else:
                            chunks.append(json.dumps(current, indent=2))
            except:
                # Fallback
                chunks.append(content[:min(1000, len(content))])
        
        # XML with XPath-like expression
        elif content.strip().startswith('<') and '>' in content:
            # Very basic implementation
            if element_path.startswith('//'):
                element = element_path[2:]
                # Simple tag search
                pattern = f'<{element}[^>]*>(.*?)</{element}>'
                matches = re.findall(pattern, content, re.DOTALL)
                
                for match in matches[:5]:  # Limit to 5 matches
                    chunks.append(f'<{element}>{match}</{element}>')
            
            if not chunks:
                # Fallback
                chunks.append(content[:min(1000, len(content))])
        
        # Fallback if no chunks were created
        if not chunks:
            chunks = self._preview_semi_structured_elements(content)
        
        return chunks
    
    def _preview_unstructured_fixed_size(self, content: str, chunk_size: int = 1000, 
                                      chunk_overlap: int = 200) -> List[str]:
        """Generate preview chunks for unstructured text using fixed size chunking"""
        chunks = []
        
        # Simple fixed-size chunking with overlap
        for i in range(0, len(content), chunk_size - chunk_overlap):
            end = min(i + chunk_size, len(content))
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence ending punctuation followed by space or newline
                for j in range(end, max(i, end - 100), -1):
                    if j < len(content) and content[j] in '.!?' and (j + 1 == len(content) or content[j + 1].isspace()):
                        end = j + 1
                        break
            
            chunk = content[i:end]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
                
            # Only generate a few chunks for preview
            if len(chunks) >= 3:
                break
        
        return chunks
    
    def _preview_unstructured_sentence_based(self, content: str) -> List[str]:
        """Generate preview chunks for unstructured text using sentence-based chunking"""
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed a reasonable chunk size
            if current_chunk and (current_length + sentence_length > 1000):
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
                
                # Only generate a few chunks for preview
                if len(chunks) >= 3:
                    break
            
            # Add the sentence to the current chunk
            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for the space
        
        # Add the final chunk if there's anything left
        if current_chunk and len(chunks) < 3:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _preview_unstructured_paragraph_based(self, content: str) -> List[str]:
        """Generate preview chunks for unstructured text using paragraph-based chunking"""
        # Split text into paragraphs (delimiter: empty line)
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', content)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            paragraph_clean = paragraph.strip()
            if not paragraph_clean:
                continue
                
            paragraph_length = len(paragraph_clean)
            
            # If adding this paragraph would exceed a reasonable chunk size
            if current_chunk and (current_length + paragraph_length > 1000):
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
                
                # Only generate a few chunks for preview
                if len(chunks) >= 3:
                    break
            
            # Add the paragraph to the current chunk
            current_chunk.append(paragraph_clean)
            current_length += paragraph_length + 2  # +2 for the newlines
        
        # Add the final chunk if there's anything left
        if current_chunk and len(chunks) < 3:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _preview_unstructured_regex_based(self, content: str, regex_pattern: str) -> List[str]:
        """Generate preview chunks for unstructured text using regex-based chunking"""
        try:
            # Split text using the provided regex pattern
            chunks_raw = re.split(regex_pattern, content)
            
            # Filter out empty chunks and limit to 3 for preview
            chunks = [chunk.strip() for chunk in chunks_raw if chunk.strip()]
            return chunks[:3]
        except:
            # Fallback to paragraph-based chunking
            logger.warning(f"Error in regex chunking preview, falling back to paragraph-based")
            return self._preview_unstructured_paragraph_based(content)