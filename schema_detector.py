import re
import json
import logging
import csv
from io import StringIO
from typing import Dict, Any, List, Optional, Tuple

try:
    # For XML documents
    from lxml import etree
except ImportError:
    etree = None

logger = logging.getLogger(__name__)

class DocumentSchemaDetector:
    """
    Detects the schema/structure type of documents and provides confidence scores.
    """
    
    def __init__(self):
        """Initialize the schema detector"""
        # MIME type mappings to structure types
        self.mime_type_mappings = {
            # Structured data formats
            'text/csv': 'structured',
            'application/vnd.ms-excel': 'structured',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'structured',
            'application/vnd.oasis.opendocument.spreadsheet': 'structured',
            
            # Semi-structured data formats
            'application/json': 'semi_structured',
            'application/xml': 'semi_structured',
            'text/xml': 'semi_structured',
            'application/yaml': 'semi_structured',
            'text/yaml': 'semi_structured',
            'text/markdown': 'semi_structured',
            'text/html': 'semi_structured',
            
            # Unstructured data formats
            'text/plain': 'unstructured',
            'application/pdf': 'unstructured',
            'application/msword': 'unstructured',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'unstructured'
        }
        
        # Structure detection functions, ordered by priority
        self.detection_functions = [
            self._detect_json_structure,
            self._detect_csv_structure,
            self._detect_xml_structure,
            self._detect_markdown_structure,
            self._detect_tabular_structure,
            self._detect_unstructured_text
        ]
    
    def detect_schema(self, raw_text: str, mime_type: Optional[str] = None, 
                    override_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect the schema/structure type of a document.
        
        Args:
            raw_text: The raw text content of the document
            mime_type: Optional MIME type hint
            override_type: Optional manual override for structure type
            
        Returns:
            Dictionary with detection results, including structure type and confidence
        """
        # If override_type is provided, return it immediately with max confidence
        if override_type and override_type in ['structured', 'semi_structured', 'unstructured']:
            return {
                'detected_type': override_type,
                'confidence_score': 100.0,
                'structure_hint': 'Manual override'
            }
            
        # Start with default result
        result = {
            'detected_type': 'unstructured',
            'confidence_score': 0.0,
            'structure_hint': 'Default detection'
        }
        
        # If empty content, return default
        if not raw_text or not raw_text.strip():
            result['structure_hint'] = 'Empty content'
            return result
        
        # If MIME type is provided and recognized, use it as a hint
        if mime_type and mime_type in self.mime_type_mappings:
            mime_hint = self.mime_type_mappings[mime_type]
            initial_confidence = 70.0  # Initial confidence based on MIME type alone
            
            result = {
                'detected_type': mime_hint,
                'confidence_score': initial_confidence,
                'structure_hint': f'Based on MIME type: {mime_type}'
            }
        
        # Perform content-based detection
        best_detection = {'type': 'unstructured', 'confidence': 0.0, 'hint': ''}
        
        # Try each detection function
        for detect_fn in self.detection_functions:
            detection = detect_fn(raw_text)
            
            # If we have a high confidence detection (>80%), use it immediately
            if detection['confidence'] > 80:
                best_detection = detection
                break
                
            # Otherwise keep track of the highest confidence detection
            if detection['confidence'] > best_detection['confidence']:
                best_detection = detection
        
        # If content-based detection has higher confidence than MIME type hint, use it
        if best_detection['confidence'] > result['confidence_score']:
            result = {
                'detected_type': best_detection['type'],
                'confidence_score': best_detection['confidence'],
                'structure_hint': best_detection['hint']
            }
        
        return result
    
    def _detect_json_structure(self, content: str) -> Dict[str, Any]:
        """Detect if content is JSON structured data"""
        result = {'type': 'unstructured', 'confidence': 0.0, 'hint': ''}
        
        # Check if content starts with typical JSON characters
        content_trimmed = content.strip()
        if content_trimmed.startswith('{') and content_trimmed.endswith('}'):
            try:
                json_data = json.loads(content)
                # Successfully parsed as JSON object
                result = {
                    'type': 'semi_structured',
                    'confidence': 95.0,
                    'hint': 'Valid JSON object detected'
                }
                return result
            except:
                # Looks like JSON but couldn't be parsed
                result = {
                    'type': 'semi_structured', 
                    'confidence': 60.0,
                    'hint': 'Content appears to be JSON-like but has syntax errors'
                }
                
        elif content_trimmed.startswith('[') and content_trimmed.endswith(']'):
            try:
                json_data = json.loads(content)
                # Check if it's an array of objects (structured)
                if json_data and isinstance(json_data, list) and isinstance(json_data[0], dict):
                    # Count object keys to see if they're consistent
                    first_keys = set(json_data[0].keys())
                    consistent_keys = True
                    
                    # Check a sample of the objects
                    sample_size = min(10, len(json_data))
                    for i in range(1, sample_size):
                        if set(json_data[i].keys()) != first_keys:
                            consistent_keys = False
                            break
                    
                    if consistent_keys:
                        # Consistent array of objects
                        result = {
                            'type': 'structured',
                            'confidence': 90.0,
                            'hint': 'JSON array of objects with consistent keys detected'
                        }
                    else:
                        # Array of objects with varying keys
                        result = {
                            'type': 'semi_structured',
                            'confidence': 85.0,
                            'hint': 'JSON array of objects with varying keys detected'
                        }
                elif isinstance(json_data, list):
                    # Array of primitives or mixed types
                    result = {
                        'type': 'semi_structured',
                        'confidence': 80.0,
                        'hint': 'JSON array of primitive values detected'
                    }
                return result
            except:
                # Looks like JSON array but couldn't be parsed
                result = {
                    'type': 'semi_structured', 
                    'confidence': 60.0,
                    'hint': 'Content appears to be JSON-like array but has syntax errors'
                }
        
        # Look for JSON-like patterns even if not valid JSON
        json_object_pattern = r'\{\s*"[^"]+"\s*:\s*(?:"[^"]*"|null|true|false|\d+(?:\.\d+)?)'
        json_array_pattern = r'\[\s*(?:"[^"]*"|null|true|false|\d+(?:\.\d+)?)'
        
        if re.search(json_object_pattern, content_trimmed):
            result = {
                'type': 'semi_structured', 
                'confidence': max(result['confidence'], 50.0),
                'hint': 'Content contains JSON-like key-value patterns'
            }
        elif re.search(json_array_pattern, content_trimmed):
            result = {
                'type': 'semi_structured', 
                'confidence': max(result['confidence'], 40.0),
                'hint': 'Content contains JSON-like array patterns'
            }
            
        return result
    
    def _detect_csv_structure(self, content: str) -> Dict[str, Any]:
        """Detect if content is CSV structured data"""
        result = {'type': 'unstructured', 'confidence': 0.0, 'hint': ''}
        
        # If content is too short, it's unlikely to be CSV
        if len(content) < 10:
            return result
            
        # Sample the content (first few lines)
        sample_lines = content.split('\n', 10)[:10]
        sample = '\n'.join(sample_lines)
        
        # Try to detect CSV dialect and structure
        try:
            # Use CSV Sniffer to detect dialect
            dialect = csv.Sniffer().sniff(sample)
            has_header = csv.Sniffer().has_header(sample)
            
            # Now try to parse the content
            f = StringIO(sample)
            reader = csv.reader(f, dialect)
            rows = list(reader)
            
            # Must have at least two rows (or one row and a header)
            if len(rows) >= 2 or (has_header and len(rows) >= 1):
                # Check consistency of column count
                first_row_len = len(rows[0])
                consistent_columns = all(len(row) == first_row_len for row in rows)
                
                if consistent_columns and first_row_len > 1:
                    # Good CSV structure
                    result = {
                        'type': 'structured',
                        'confidence': 90.0,
                        'hint': f'CSV with {first_row_len} columns detected'
                    }
                elif first_row_len > 1:
                    # Inconsistent column counts
                    result = {
                        'type': 'structured',
                        'confidence': 75.0,
                        'hint': 'CSV with varying column counts detected'
                    }
                else:
                    # Single column "CSV" is more likely just text
                    result = {
                        'type': 'unstructured',
                        'confidence': 60.0,
                        'hint': 'Single-column CSV detected, likely regular text'
                    }
        except:
            # Fallback detection based on patterns
            comma_split_rows = [len(line.split(',')) for line in sample_lines if line.strip()]
            tab_split_rows = [len(line.split('\t')) for line in sample_lines if line.strip()]
            semicolon_split_rows = [len(line.split(';')) for line in sample_lines if line.strip()]
            
            # Check for consistent number of columns with common delimiters
            for delimiter, split_rows, name in [
                (',', comma_split_rows, 'comma'), 
                ('\t', tab_split_rows, 'tab'),
                (';', semicolon_split_rows, 'semicolon')
            ]:
                if split_rows and all(r == split_rows[0] for r in split_rows) and split_rows[0] > 1:
                    # Consistent columns with this delimiter
                    result = {
                        'type': 'structured',
                        'confidence': 70.0,
                        'hint': f'Possible {name}-delimited data with {split_rows[0]} columns'
                    }
                    break
        
        return result
    
    def _detect_xml_structure(self, content: str) -> Dict[str, Any]:
        """Detect if content is XML/HTML structured data"""
        result = {'type': 'unstructured', 'confidence': 0.0, 'hint': ''}
        
        # Check if content has XML/HTML structure
        content_trimmed = content.strip()
        if content_trimmed.startswith('<') and content_trimmed.endswith('>'):
            # Look for opening and closing tags
            if re.search(r'<([a-zA-Z][a-zA-Z0-9_:-]*)[^>]*>.*?</\1>', content, re.DOTALL):
                try:
                    if etree is not None:
                        # Try parsing with lxml
                        root = etree.fromstring(content.encode('utf-8'))
                        # Successfully parsed as XML
                        
                        # Check for repeated elements that might indicate structured data
                        element_counts = {}
                        for child in root.iter():
                            tag = child.tag
                            if tag not in element_counts:
                                element_counts[tag] = 0
                            element_counts[tag] += 1
                        
                        # If we have elements that repeat many times, it's more likely structured
                        repeated_elements = [tag for tag, count in element_counts.items() if count >= 5]
                        
                        if repeated_elements:
                            result = {
                                'type': 'structured',
                                'confidence': 80.0,
                                'hint': f'XML with repeated elements detected ({repeated_elements[0]})'
                            }
                        else:
                            result = {
                                'type': 'semi_structured',
                                'confidence': 90.0,
                                'hint': 'Valid XML document detected'
                            }
                    else:
                        # lxml not available, rely on regex detection
                        result = {
                            'type': 'semi_structured',
                            'confidence': 80.0,
                            'hint': 'XML-like document detected'
                        }
                except:
                    # Looks like XML but couldn't be parsed
                    result = {
                        'type': 'semi_structured', 
                        'confidence': 70.0,
                        'hint': 'Content appears to be XML-like but may have syntax errors'
                    }
            else:
                # Has < > tags but not properly matched, could be malformed HTML
                result = {
                    'type': 'semi_structured', 
                    'confidence': 50.0,
                    'hint': 'Content contains XML/HTML tags but not properly matched'
                }
        
        # Check for HTML specifically
        if re.search(r'<!DOCTYPE html>|<html[^>]*>|<head[^>]*>|<body[^>]*>', content, re.IGNORECASE):
            result = {
                'type': 'semi_structured',
                'confidence': 85.0,
                'hint': 'HTML document detected'
            }
            
        return result
    
    def _detect_markdown_structure(self, content: str) -> Dict[str, Any]:
        """Detect if content is Markdown structured data"""
        result = {'type': 'unstructured', 'confidence': 0.0, 'hint': ''}
        
        # Look for Markdown headers
        header_matches = re.findall(r'^(#+)\s+(.+)$', content, re.MULTILINE)
        
        if header_matches:
            # Count headers by level
            header_levels = [len(level) for level, _ in header_matches]
            
            # If we have headers with multiple levels, it's more structured
            if len(set(header_levels)) > 1:
                result = {
                    'type': 'semi_structured',
                    'confidence': 80.0,
                    'hint': f'Markdown with {len(header_matches)} headers across {len(set(header_levels))} levels'
                }
            else:
                result = {
                    'type': 'semi_structured',
                    'confidence': 70.0,
                    'hint': f'Markdown with {len(header_matches)} headers detected'
                }
                
        # Look for other Markdown elements
        if re.search(r'\[.+?\]\(.+?\)', content):  # Links
            result['confidence'] = max(result['confidence'], 65.0)
            if not result['hint']:
                result['type'] = 'semi_structured'
                result['hint'] = 'Markdown with links detected'
                
        if re.search(r'^\s*[*+-]\s+', content, re.MULTILINE):  # Lists
            result['confidence'] = max(result['confidence'], 65.0)
            if not result['hint']:
                result['type'] = 'semi_structured'
                result['hint'] = 'Markdown with lists detected'
                
        if re.search(r'^```.*?```', content, re.MULTILINE | re.DOTALL):  # Code blocks
            result['confidence'] = max(result['confidence'], 70.0)
            if not result['hint']:
                result['type'] = 'semi_structured'
                result['hint'] = 'Markdown with code blocks detected'
                
        if re.search(r'^\|.*\|$', content, re.MULTILINE):  # Tables
            table_rows = re.findall(r'^\|.*\|$', content, re.MULTILINE)
            if len(table_rows) >= 3:  # Header, separator, data
                result['type'] = 'semi_structured'
                result['confidence'] = max(result['confidence'], 75.0)
                if not result['hint'] or 'headers' in result['hint']:
                    result['hint'] = 'Markdown with tables detected'
                    
        return result
    
    def _detect_tabular_structure(self, content: str) -> Dict[str, Any]:
        """Detect if content has any other tabular structure"""
        result = {'type': 'unstructured', 'confidence': 0.0, 'hint': ''}
        
        # Look for table-like patterns that aren't CSV, JSON, or other known formats
        
        # Check for fixed-width columns
        lines = content.split('\n')[:20]  # Sample the first 20 lines
        
        if len(lines) >= 5:  # Need at least a few lines to detect
            # Check for lines with consistent spacing or separators
            whitespace_pattern = re.compile(r'\s{2,}')
            
            # Count whitespace stretches in each line
            whitespace_counts = []
            consistent_whitespace = True
            
            for line in lines:
                if not line.strip():
                    continue
                    
                whitespace_stretches = whitespace_pattern.findall(line)
                whitespace_counts.append(len(whitespace_stretches))
                
                # Check alignment of whitespace positions
                if len(whitespace_counts) >= 2:
                    if whitespace_counts[-1] != whitespace_counts[-2]:
                        consistent_whitespace = False
                        break
            
            if consistent_whitespace and whitespace_counts and whitespace_counts[0] >= 2:
                # Lines have consistent whitespace patterns, suggesting fixed-width columns
                result = {
                    'type': 'structured',
                    'confidence': 65.0,
                    'hint': f'Fixed-width columns detected with {whitespace_counts[0] + 1} columns'
                }
        
        return result
    
    def _detect_unstructured_text(self, content: str) -> Dict[str, Any]:
        """Detect unstructured text with confidence based on patterns"""
        result = {'type': 'unstructured', 'confidence': 0.0, 'hint': ''}
        
        # Count paragraphs (text blocks separated by blank lines)
        paragraphs = re.split(r'\n\s*\n', content)
        paragraphs = [p for p in paragraphs if p.strip()]
        
        if paragraphs:
            # Text with multiple paragraphs
            avg_para_length = sum(len(p) for p in paragraphs) / len(paragraphs)
            
            if len(paragraphs) >= 3 and avg_para_length > 100:
                # Longer paragraphs suggest narrative text
                result = {
                    'type': 'unstructured',
                    'confidence': 85.0,
                    'hint': f'Narrative text with {len(paragraphs)} paragraphs detected'
                }
            elif len(paragraphs) >= 2:
                # Multiple paragraphs of any length
                result = {
                    'type': 'unstructured',
                    'confidence': 75.0,
                    'hint': 'Multiple text paragraphs detected'
                }
            else:
                # Single paragraph
                result = {
                    'type': 'unstructured',
                    'confidence': 65.0,
                    'hint': 'Single text paragraph detected'
                }
                
        # Look for sentence patterns
        sentence_count = len(re.findall(r'[.!?]\s+[A-Z]', content))
        
        if sentence_count >= 5:
            # Multiple sentences with proper capitalization
            result = {
                'type': 'unstructured',
                'confidence': max(result['confidence'], 80.0),
                'hint': 'Multiple well-formed sentences detected'
            }
            
        return result