import os
import re
import json
import logging
from typing import Dict, Optional, Union, Tuple, List, Any

import pandas as pd
import yaml
import xml.etree.ElementTree as ET
from lxml import etree

logger = logging.getLogger(__name__)

class DocumentSchemaDetector:
    """
    Detects the schema/structure type of an uploaded document to guide further processing.
    
    This class can determine if a document contains structured data (like CSV, Excel),
    semi-structured data (like JSON, XML, YAML, Markdown), or unstructured data (plain text).
    """
    
    # Constants for document types
    STRUCTURED = "structured"
    SEMI_STRUCTURED = "semi_structured"
    UNSTRUCTURED = "unstructured"
    
    # File extension mappings
    EXTENSION_MAPPINGS = {
        # Structured data formats
        'csv': STRUCTURED,
        'tsv': STRUCTURED,
        'xlsx': STRUCTURED,
        'xls': STRUCTURED,
        'ods': STRUCTURED,
        
        # Semi-structured data formats
        'json': SEMI_STRUCTURED,
        'xml': SEMI_STRUCTURED,
        'yaml': SEMI_STRUCTURED,
        'yml': SEMI_STRUCTURED,
        'md': SEMI_STRUCTURED,
        'html': SEMI_STRUCTURED,
        
        # Typically unstructured (but could be either)
        'txt': None,  # Need content analysis
        'pdf': None,  # Need content analysis
        'doc': None,  # Need content analysis
        'docx': None,  # Need content analysis
        'rtf': None,  # Need content analysis
    }
    
    # Chunking recommendations based on document type
    CHUNKING_RECOMMENDATIONS = {
        STRUCTURED: {
            "method": "row_based",
            "description": "Split by rows with header context preservation"
        },
        SEMI_STRUCTURED: {
            "method": "semantic_hierarchy",
            "description": "Split by semantic elements (JSON objects, XML nodes)"
        },
        UNSTRUCTURED: {
            "method": "paragraph_or_sentence",
            "description": "Split by paragraphs with overlap"
        }
    }
    
    def __init__(self, sample_size: int = 5000):
        """
        Initialize the schema detector.
        
        Args:
            sample_size: Maximum number of characters to analyze for detection
        """
        self.sample_size = sample_size
    
    def detect_schema(self, 
                     file_path: Optional[str] = None, 
                     raw_text: Optional[str] = None,
                     mime_type: Optional[str] = None,
                     override_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect the schema type of a document.
        
        Args:
            file_path: Path to the document file
            raw_text: Raw text content of the document
            mime_type: MIME type of the document if known
            override_type: Manually override the detected type
            
        Returns:
            Dictionary with detection results:
            {
                "detected_type": One of ["structured", "semi_structured", "unstructured"],
                "confidence_score": Percentage confidence in the detection,
                "recommendation": Chunking recommendation,
                "structure_hint": Example of detected structure (if applicable)
            }
        """
        # Manual override if specified
        if override_type and override_type in [self.STRUCTURED, self.SEMI_STRUCTURED, self.UNSTRUCTURED]:
            return self._create_result(override_type, 100, structure_hint="Manual override")
        
        # Validate inputs
        if not file_path and not raw_text:
            raise ValueError("Either file_path or raw_text must be provided")
        
        # Load content if file_path is provided
        if file_path:
            file_extension = os.path.splitext(file_path)[1].lower().lstrip('.')
            
            # Check if we can determine type from extension
            if file_extension in self.EXTENSION_MAPPINGS and self.EXTENSION_MAPPINGS[file_extension]:
                detected_type = self.EXTENSION_MAPPINGS[file_extension]
                confidence = 90  # High confidence based on extension
                structure_hint = self._extract_structure_hint(file_path, detected_type)
                return self._create_result(detected_type, confidence, structure_hint)
            
            # Otherwise, we need to analyze content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read(self.sample_size)
            except UnicodeDecodeError:
                # Binary file or non-UTF-8 encoding
                return self._create_result(self.UNSTRUCTURED, 60, 
                                          structure_hint="Binary or non-UTF-8 file detected")
        else:
            # Use the provided raw text
            content = raw_text[:self.sample_size] if raw_text else ""
        
        # Analyze the content to determine structure
        return self._analyze_content(content, mime_type)
    
    def _analyze_content(self, content: str, mime_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze content to determine its structure type.
        
        Args:
            content: Text content to analyze
            mime_type: MIME type if known
            
        Returns:
            Detection result dictionary
        """
        # Check common patterns and formats
        
        # Check for structured data (CSV-like)
        csv_score, csv_hint = self._check_csv_pattern(content)
        
        # Check for JSON
        json_score, json_hint = self._check_json_pattern(content)
        
        # Check for XML/HTML
        xml_score, xml_hint = self._check_xml_pattern(content)
        
        # Check for YAML
        yaml_score, yaml_hint = self._check_yaml_pattern(content)
        
        # Check for Markdown
        markdown_score, markdown_hint = self._check_markdown_pattern(content)
        
        # Determine the highest scoring format
        format_scores = [
            (self.STRUCTURED, csv_score, csv_hint),
            (self.SEMI_STRUCTURED, max(json_score, xml_score, yaml_score, markdown_score),
             json_hint if json_score > max(xml_score, yaml_score, markdown_score) else
             xml_hint if xml_score > max(yaml_score, markdown_score) else
             yaml_hint if yaml_score > markdown_score else markdown_hint)
        ]
        
        # If no strong pattern is detected, default to unstructured
        max_score = max(score for _, score, _ in format_scores)
        if max_score < 30:
            return self._create_result(
                self.UNSTRUCTURED, 
                70, 
                structure_hint="No clear structure patterns detected"
            )
        
        # Get the highest scoring format
        detected_type, score, hint = max(format_scores, key=lambda x: x[1])
        
        # Return the result
        return self._create_result(detected_type, score, hint)
    
    def _check_csv_pattern(self, content: str) -> Tuple[float, Optional[str]]:
        """Check if content matches CSV/tabular patterns"""
        lines = content.strip().split('\n')
        if len(lines) < 2:
            return 0, None
        
        # Look for consistent delimiters
        potential_delimiters = [',', '\t', '|', ';']
        delimiter_consistency = {}
        
        for delimiter in potential_delimiters:
            if delimiter not in content:
                continue
                
            # Check if number of delimiters is consistent across lines
            counts = [line.count(delimiter) for line in lines[:10]]
            if len(counts) >= 2 and all(c > 0 for c in counts):
                # Calculate consistency (0 to 1)
                consistency = sum(1 for i in range(1, len(counts)) 
                               if counts[i] == counts[0]) / (len(counts) - 1)
                delimiter_consistency[delimiter] = consistency
        
        if not delimiter_consistency:
            return 0, None
            
        best_delimiter = max(delimiter_consistency.items(), key=lambda x: x[1])
        
        if best_delimiter[1] > 0.7:  # If 70%+ consistency
            # Extract potential header
            header = lines[0].split(best_delimiter[0])
            confidence = min(best_delimiter[1] * 100, 95)  # Cap at 95%
            return confidence, f"Header: {header[:3]}{'...' if len(header) > 3 else ''}"
            
        return 0, None
    
    def _check_json_pattern(self, content: str) -> Tuple[float, Optional[str]]:
        """Check if content matches JSON patterns"""
        # Quick check for JSON markers
        content = content.strip()
        json_object_pattern = content.startswith('{') and content.endswith('}')
        json_array_pattern = content.startswith('[') and content.endswith(']')
        
        # More lenient check for partial content
        has_json_syntax = ('{' in content and '}' in content) or ('[' in content and ']' in content)
        has_quotes = '"' in content
        has_colons = ':' in content
        
        # If it doesn't even have basic JSON syntax, return early
        if not (json_object_pattern or json_array_pattern) and not (has_json_syntax and has_quotes and has_colons):
            return 0, None
        
        try:
            # Try to parse as JSON
            data = json.loads(content)
            
            # Extract structure hint
            if isinstance(data, dict):
                keys = list(data.keys())[:3]
                hint = f"Keys: {keys}{'...' if len(data) > 3 else ''}"
                return 95, hint
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                keys = list(data[0].keys())[:3]
                hint = f"Array of objects with keys: {keys}{'...' if len(data[0]) > 3 else ''}"
                return 95, hint
            elif isinstance(data, list):
                return 90, f"JSON array with {len(data)} items"
            else:
                return 90, "Valid JSON"
        except json.JSONDecodeError:
            # More careful analysis for partial JSON
            if not (json_object_pattern or json_array_pattern) and has_json_syntax:
                # Look for patterns like "key": value pairs
                key_val_pattern = re.compile(r'"[^"]+"\s*:\s*("[^"]*"|[\d\.]+|\{|\[|true|false|null)')
                matches = key_val_pattern.findall(content)
                
                if matches:
                    confidence = min(50 + len(matches) * 5, 80)  # More matches = higher confidence
                    return confidence, "Partial JSON with key-value pairs"
            
            # Even if it doesn't fully parse, if it has all markers it's likely JSON
            if (json_object_pattern or json_array_pattern) and has_quotes and has_colons:
                return 75, "Malformed but likely JSON"
                
            return 40, "Potential JSON-like content"
    
    def _check_xml_pattern(self, content: str) -> Tuple[float, Optional[str]]:
        """Check if content matches XML/HTML patterns"""
        content = content.strip()
        
        # Quick check for XML markers
        if content.startswith('<?xml') or content.startswith('<') and '>' in content:
            try:
                # Try to parse as XML
                root = ET.fromstring(content)
                tag_name = root.tag
                if tag_name.startswith('{'):  # Handle namespaces
                    tag_name = tag_name.split('}')[-1]
                    
                children = list(root)
                if children:
                    child_tags = [child.tag.split('}')[-1] if child.tag.startswith('{') else child.tag 
                                 for child in children[:3]]
                    hint = f"Root: <{tag_name}> with children: {child_tags}"
                else:
                    hint = f"Root element: <{tag_name}>"
                return 95, hint
            except ET.ParseError:
                # Check for HTML
                if '<html' in content.lower() or '<body' in content.lower() or '<div' in content.lower():
                    try:
                        # Try to parse with lxml which is more forgiving
                        parser = etree.HTMLParser()
                        etree.fromstring(content, parser)
                        return 85, "HTML document"
                    except Exception:
                        pass
                return 50, "Partial or malformed XML/HTML"
        return 0, None
    
    def _check_yaml_pattern(self, content: str) -> Tuple[float, Optional[str]]:
        """Check if content matches YAML patterns"""
        # Quick pattern check for YAML
        if not re.search(r'^[a-zA-Z0-9_-]+:\s', content, re.MULTILINE):
            return 0, None
            
        try:
            # Try to parse as YAML
            data = yaml.safe_load(content)
            
            if isinstance(data, dict):
                keys = list(data.keys())[:3]
                hint = f"Keys: {keys}{'...' if len(data) > 3 else ''}"
                return 90, hint
            else:
                return 70, "Valid YAML (non-object)"
        except yaml.YAMLError:
            # Check for common YAML patterns
            indent_pattern = re.compile(r'^[ ]{2,}[a-zA-Z0-9_-]+:', re.MULTILINE)
            list_pattern = re.compile(r'^[ ]*-[ ]+', re.MULTILINE)
            
            if indent_pattern.search(content) or list_pattern.search(content):
                return 50, "Partial or malformed YAML"
            
            return 0, None
    
    def _check_markdown_pattern(self, content: str) -> Tuple[float, Optional[str]]:
        """Check if content matches Markdown patterns"""
        # Check for common Markdown patterns
        patterns = [
            (r'^#+ ', 'Headers'),                 # Headers
            (r'[*_]{1,2}[^*_]+[*_]{1,2}', 'Emphasis'),  # Bold/Italic
            (r'^- ', 'Lists'),                    # Unordered lists
            (r'^[0-9]+\. ', 'Numbered lists'),    # Ordered lists
            (r'!\[.*?\]\(.*?\)', 'Images'),       # Images
            (r'\[.*?\]\(.*?\)', 'Links'),         # Links
            (r'^```', 'Code blocks'),             # Code blocks
            (r'^>', 'Blockquotes'),               # Blockquotes
            (r'^\|.*\|$', 'Tables')               # Tables
        ]
        
        matches = []
        for pattern, name in patterns:
            if re.search(pattern, content, re.MULTILINE):
                matches.append(name)
        
        if matches:
            score = min(len(matches) * 15, 90)  # 15 points per match, max 90
            return score, f"Markdown elements: {', '.join(matches[:3])}"
        
        return 0, None
    
    def _extract_structure_hint(self, file_path: str, doc_type: str) -> Optional[str]:
        """
        Extract a hint about the document's structure based on file type.
        
        Args:
            file_path: Path to the document
            doc_type: Detected document type
            
        Returns:
            String hint about document structure, or None if unavailable
        """
        try:
            extension = os.path.splitext(file_path)[1].lower()
            
            if doc_type == self.STRUCTURED:
                if extension in ['.csv', '.tsv']:
                    df = pd.read_csv(file_path, nrows=1)
                    return f"CSV columns: {list(df.columns)[:5]}"
                elif extension in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path, nrows=1)
                    return f"Excel columns: {list(df.columns)[:5]}"
            
            elif doc_type == self.SEMI_STRUCTURED:
                if extension == '.json':
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        return f"JSON keys: {list(data.keys())[:5]}"
                    elif isinstance(data, list) and data and isinstance(data[0], dict):
                        return f"JSON array with keys: {list(data[0].keys())[:5]}"
                
                elif extension in ['.xml', '.html']:
                    tree = ET.parse(file_path)
                    root = tree.getroot()
                    return f"XML root: {root.tag}"
                
                elif extension in ['.yaml', '.yml']:
                    with open(file_path, 'r') as f:
                        data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        return f"YAML keys: {list(data.keys())[:5]}"
        
        except Exception as e:
            logger.warning(f"Could not extract structure hint: {str(e)}")
            
        return None
    
    def _create_result(self, 
                      detected_type: str, 
                      confidence: float, 
                      structure_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a standardized result dictionary.
        
        Args:
            detected_type: The detected document type
            confidence: Confidence score (0-100)
            structure_hint: Optional hint about the document structure
            
        Returns:
            Result dictionary
        """
        recommendation = self.CHUNKING_RECOMMENDATIONS.get(detected_type, 
                                                          self.CHUNKING_RECOMMENDATIONS[self.UNSTRUCTURED])
        
        result = {
            "detected_type": detected_type,
            "confidence_score": round(confidence, 2),
            "recommendation": recommendation,
        }
        
        if structure_hint:
            result["structure_hint"] = structure_hint
            
        return result