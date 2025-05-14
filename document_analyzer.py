# Adding samples and descriptions for JSON and CSV fields in document_analyzer.py

import re
import logging
import json
from typing import Dict, Any, List, Optional
from app import db
from models import Document, Chunk, Metadata
from document_processor import DocumentProcessor
from schema_detector import DocumentSchemaDetector

logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    def __init__(self, document_processor: DocumentProcessor):
        self.document_processor = document_processor
        self.schema_detector = DocumentSchemaDetector()

    def analyze_document(self, document_id: str) -> Dict[str, Any]:
        document = Document.query.get_or_404(document_id)
        metadata = {m.key: m.value for m in Metadata.query.filter_by(document_id=document_id).all()}

        # Get document content
        content = document.content

        # Detect structure if not already set
        if 'detected_structure' not in metadata:
            structure_info = self.schema_detector.detect_schema(content)
            metadata.update({
                'detected_structure': structure_info['detected_type'],
                'detection_confidence': str(structure_info['confidence_score']),
                'structure_hint': structure_info.get('structure_hint', '')
            })
            self._save_metadata(document_id, metadata)

        structure_type = metadata.get('detected_structure', 'unstructured')

        # Extract fields based on structure type
        fields = []
        try:
            if structure_type == 'structured':
                fields = self._extract_csv_fields(content)
            elif structure_type == 'semi_structured':
                if content.strip().startswith('{') or content.strip().startswith('['):
                    try:
                        json_data = json.loads(content)
                        fields = self._extract_json_fields(json_data)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parsing error: {e}")
                else:
                    fields = self._extract_xml_fields(content)

            # Log detected fields
            logger.info(f"Detected {len(fields)} fields for document {document_id}")
        except Exception as e:
            logger.error(f"Error extracting fields: {e}")
            fields = []

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

        # Load existing chunking config
        try:
            if 'chunking_config' in metadata:
                saved_config = json.loads(metadata['chunking_config'])
                # Update entire config dictionary
                for structure in chunking_config:
                    if structure in saved_config:
                        chunking_config[structure].update(saved_config[structure])
        except:
            logger.warning("Failed to load existing chunking config")

        # Initialize embedding configuration with defaults
        embedding_config = {
            'dtype': 'float32',
            'normalize': True,
            'max_tokens_per_chunk': 512
        }

        # Load existing embedding config
        try:
            if 'embedding_config' in metadata:
                saved_config = json.loads(metadata['embedding_config'])
                embedding_config.update(saved_config)
        except:
            logger.warning("Failed to load existing embedding config")

        # Extract custom metadata
        custom_metadata = {}
        reserved_keys = {'detected_structure', 'detection_confidence', 'chunking_config', 
                        'embedding_config', 'store_full_content', 'language'}
        for key, value in metadata.items():
            if key not in reserved_keys:
                custom_metadata[key] = value

        # Suggest fields for embedding and metadata
        suggested_fields = self._suggest_fields(fields, structure_type)

        return {
            'document': document,
            'metadata': metadata,
            'fields': fields,
            'structure_type': structure_type,
            'custom_metadata': custom_metadata,
            'store_full_content': metadata.get('store_full_content', 'false') == 'true',
            'detected_structure_name': metadata.get('detected_structure', 'unstructured'),
            'detection_confidence': float(metadata.get('detection_confidence', '0.0')),
            'detected_language': metadata.get('language', 'ar'),
            'chunking_config': chunking_config,
            'embedding_config': embedding_config,
            'suggested_embedding_fields': suggested_fields['embedding'],
            'suggested_metadata_fields': suggested_fields['metadata']
        }

    def _extract_json_fields(self, data: Any, parent_path: str = '', max_items: int = 5) -> List[Dict[str, Any]]:
        """Extract field names, types, and samples from JSON data"""
        fields = []

        def get_sample_value(value: Any) -> str:
            """Get a sample value suitable for display"""
            if isinstance(value, (dict, list)):
                sample = json.dumps(value, ensure_ascii=False)[:100]
                return f"{sample}..." if len(sample) == 100 else sample
            else:
                str_val = str(value)
                return f"{str_val[:100]}..." if len(str_val) > 100 else str_val

        def get_field_description(value: Any) -> str:
            """Get a field description based on type and content"""
            if isinstance(value, dict):
                return f"كائن JSON يحتوي على {len(value)} حقل"
            elif isinstance(value, list):
                return f"مصفوفة JSON تحتوي على {len(value)} عنصر"
            elif isinstance(value, str):
                return "حقل نصي"
            elif isinstance(value, bool):
                return "حقل منطقي"
            elif isinstance(value, (int, float)):
                return "حقل رقمي"
            elif value is None:
                return "حقل فارغ"
            return f"حقل من نوع {type(value).__name__}"

        if isinstance(data, dict):
            for key, value in data.items():
                full_path = f"{parent_path}.{key}" if parent_path else key
                field_type = type(value).__name__
                fields.append({
                    'name': full_path,
                    'type': field_type,
                    'sample': get_sample_value(value),
                    'description': get_field_description(value)
                })

                if isinstance(value, (dict, list)):
                    fields.extend(self._extract_json_fields(value, full_path))

        elif isinstance(data, list) and data:
            processed_items = data[:max_items]
            for item in processed_items:
                fields.extend(self._extract_json_fields(item, parent_path))

            seen = set()
            unique_fields = []
            for field in fields:
                field_key = (field['name'], field['type'])
                if field_key not in seen:
                    seen.add(field_key)
                    unique_fields.append(field)
            fields = unique_fields

        return fields

    def _extract_csv_fields(self, content: str) -> List[Dict[str, Any]]:
        """Extract fields from CSV content with samples"""
        import csv
        from io import StringIO

        fields = []
        try:
            csv_reader = csv.reader(StringIO(content))
            headers = next(csv_reader)

            # Read sample rows
            sample_rows = []
            for _ in range(3):
                try:
                    sample_rows.append(next(csv_reader))
                except StopIteration:
                    break

            for i, header in enumerate(headers):
                field_type = self._infer_csv_field_type(sample_rows, i)

                # Get samples for this column
                samples = []
                for row in sample_rows:
                    if len(row) > i:
                        samples.append(row[i])

                # Create sample string
                sample_str = ", ".join(f"'{s}'" for s in samples[:2]) if samples else "لا يوجد عينة"
                if len(samples) > 2:
                    sample_str += "..."

                fields.append({
                    'name': header,
                    'type': field_type,
                    'sample': sample_str,
                    'description': f"عمود CSV من نوع {field_type}"
                })
        except Exception as e:
            logger.warning(f"Failed to extract CSV fields: {e}")

        return fields

    def _extract_xml_fields(self, content: str) -> List[Dict[str, str]]:
        """Extract fields from XML content"""
        from xml.etree import ElementTree as ET

        fields = []
        try:
            root = ET.fromstring(content)
            fields.extend(self._process_xml_element(root))
        except:
            logger.warning("Failed to extract XML fields")

        return fields

    def _process_xml_element(self, element: Any, parent_path: str = '') -> List[Dict[str, Any]]:
        """Process XML element recursively with namespace handling and samples"""
        fields = []

        # Handle namespace
        tag = element.tag
        if '}' in tag:
            namespace, tag = tag.split('}', 1)
            namespace = namespace[1:]  # Remove {
        else:
            namespace = None

        # Build path
        current_path = f"{parent_path}/{tag}" if parent_path else tag

        # Add current element with sample value
        fields.append({
            'name': current_path,
            'type': 'element',
            'namespace': namespace,
            'sample': element.text.strip() if element.text and element.text.strip() else None
        })

        # Add attributes with samples
        for attr, value in element.attrib.items():
            attr_path = f"{current_path}/@{attr}"
            fields.append({
                'name': attr_path,
                'type': 'attribute',
                'namespace': namespace,
                'sample': value
            })

        # Process children
        for child in element:
            fields.extend(self._process_xml_element(child, current_path))

        return fields

    def _infer_csv_field_type(self, sample_rows: List[List[str]], column_index: int) -> str:
        """Infer the type of a CSV column"""
        values = [row[column_index] for row in sample_rows if len(row) > column_index]

        if not values:
            return 'string'

        # Try numeric
        try:
            all(float(v) for v in values if v.strip())
            return 'number'
        except:
            pass

        # Try date
        import dateutil.parser
        try:
            all(dateutil.parser.parse(v) for v in values if v.strip())
            return 'date'
        except:
            return 'string'

    def _suggest_fields(self, fields: List[Dict[str, str]], structure_type: str) -> Dict[str, List[str]]:
        """Suggest fields for embedding and metadata"""
        embedding_fields = []
        metadata_fields = []

        for field in fields:
            name = field['name']
            field_type = field['type']

            # Suggest text fields for embedding
            if field_type in ('string', 'element') and not name.endswith('/@'):
                embedding_fields.append(name)

            # Suggest metadata fields
            if field_type in ('number', 'date', 'attribute') or name.endswith('/@'):
                metadata_fields.append(name)

        return {
            'embedding': embedding_fields,
            'metadata': metadata_fields
        }

    def _save_metadata(self, document_id: str, metadata: Dict[str, str]) -> None:
        """Save metadata to database"""
        for key, value in metadata.items():
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

    def save_processing_config(self, document_id: str, config: Dict[str, Any]) -> bool:
        """Save processing configuration"""
        try:
            # Define which keys to save
            config_keys = {
                'detected_structure', 'detection_confidence',
                'chunking_config', 'embedding_config',
                'store_full_content', 'language'
            }

            # Save only relevant configuration
            save_config = {k: v for k, v in config.items() if k in config_keys}

            # Convert complex values to JSON
            for key, value in save_config.items():
                if isinstance(value, (dict, list)):
                    save_config[key] = json.dumps(value)
                else:
                    save_config[key] = str(value)

            self._save_metadata(document_id, save_config)
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

            # Prepare clean metadata for document processor
            processing_metadata = {
                'structure_type': config['detected_structure'],
                'language': config.get('language', 'ar'),
                'chunking_config': config['chunking_config'][config['detected_structure']]
            }

            # Add custom metadata
            if 'custom_metadata' in config:
                processing_metadata.update(config['custom_metadata'])

            self.document_processor.update_document(
                document_id=document_id,
                name=document.name,
                content=document.content,
                metadata=processing_metadata
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
            structure_type = config['detected_structure']
            chunking_config = config['chunking_config'][structure_type]

            # Use document processor to generate chunks
            chunks = self.document_processor.generate_chunks(
                content=document.content,
                structure_type=structure_type,
                chunking_config=chunking_config
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