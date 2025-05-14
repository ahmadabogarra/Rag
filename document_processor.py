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
                        metadata: Optional[Dict[str, str]] = None,
                        structure_type: Optional[str] = None,
                        chunking_config: Optional[Dict[str, Any]] = None) -> Document:
        """
        Update an existing document, refresh chunks if content changed
        
        Args:
            document_id: ID of document to update
            name: New document name (optional)
            content: New document content (optional)
            metadata: New metadata key-value pairs (optional)
            structure_type: Document structure type (optional)
            chunking_config: Configuration for chunking strategy (optional)
            
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
            
            # Use structure_type from args or from metadata or default to unstructured
            if structure_type is None and metadata and 'detected_structure' in metadata:
                structure_type = metadata['detected_structure']
            elif structure_type is None:
                structure_type = "unstructured"
            
            # Create new chunks
            self._create_chunks(document_id, document.content, structure_type, chunking_config)
            
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
    
    def _create_chunks(self, document_id: str, content: str, 
                        doc_structure: str = "unstructured", 
                        chunking_config: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Split document content into chunks with specified size and overlap
        
        Args:
            document_id: Document ID
            content: Document content to chunk
            doc_structure: Document structure type ("structured", "semi_structured", "unstructured")
            chunking_config: Optional configuration for chunking strategy
            
        Returns:
            List of created Chunk objects
        """
        # Calculate positions for chunks based on document structure and config
        chunks_data = self._split_text(
            content, 
            self.chunk_size, 
            self.chunk_overlap, 
            doc_structure, 
            chunking_config
        )
        
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
        logger.info(f"Created {len(created_chunks)} chunks for document {document_id} using {doc_structure} chunking")
        return created_chunks
    
    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int, 
                     doc_structure: str = "unstructured", 
                     chunking_config: Optional[Dict[str, Any]] = None) -> List[Tuple[str, int, int]]:
        """
        Split text into chunks with specified size and overlap
        
        Args:
            text: Text to split
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            doc_structure: Document structure type (unstructured, structured, semi_structured)
            chunking_config: Optional configuration for chunking strategy
            
        Returns:
            List of tuples (chunk_text, start_position, end_position)
        """
        if not text:
            return []
        
        if chunking_config is None:
            chunking_config = {}
            
        # Handle different document structure types
        if doc_structure == "structured":
            config = chunking_config.get('structured', {})
            method = config.get('method', 'row_based')
            
            if method == 'row_based':
                min_rows = config.get('min_rows_per_chunk', 1)
                max_rows = config.get('max_rows_per_chunk', 10)
                return self._split_structured_text_row_based(text, min_rows, max_rows)
            elif method == 'multi_row':
                min_rows = config.get('min_rows_per_chunk', 1)
                max_rows = config.get('max_rows_per_chunk', 10)
                return self._split_structured_text_multi_row(text, min_rows, max_rows)
            else:
                # Default to standard structured text splitting
                return self._split_structured_text(text, chunk_size, chunk_overlap)
                
        elif doc_structure == "semi_structured":
            config = chunking_config.get('semi_structured', {})
            method = config.get('method', 'semantic_elements')
            
            if method == 'semantic_elements':
                preserve_hierarchy = config.get('preserve_hierarchy', True)
                return self._split_semi_structured_text_semantic(text, chunk_size, chunk_overlap, preserve_hierarchy)
            elif method == 'custom_path':
                element_path = config.get('element_path', '')
                preserve_hierarchy = config.get('preserve_hierarchy', True)
                return self._split_semi_structured_text_path(text, chunk_size, chunk_overlap, element_path, preserve_hierarchy)
            else:
                # Default to standard semi-structured text splitting
                return self._split_semi_structured_text(text, chunk_size, chunk_overlap)
                
        else:  # Unstructured
            config = chunking_config.get('unstructured', {})
            method = config.get('method', 'paragraph_based')
            
            if method == 'fixed_size':
                chunk_size = config.get('chunk_size', chunk_size)
                chunk_overlap = config.get('chunk_overlap', chunk_overlap)
                return self._split_unstructured_text_fixed_size(text, chunk_size, chunk_overlap)
            elif method == 'sentence_based':
                return self._split_unstructured_text_sentence(text, chunk_size, chunk_overlap)
            elif method == 'paragraph_based':
                return self._split_unstructured_text_paragraph(text, chunk_size, chunk_overlap)
            elif method == 'regex_based':
                regex_pattern = config.get('regex_pattern', r'\n\s*\n|\r\n\s*\r\n')
                return self._split_unstructured_text_regex(text, chunk_size, chunk_overlap, regex_pattern)
            else:
                # Default to standard unstructured text splitting
                return self._split_unstructured_text(text, chunk_size, chunk_overlap)
    
    def _split_unstructured_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int, int]]:
        """
        Default split for unstructured text using sentence-aware splitting
        Note: This is a fallback method, prefer using the specialized methods
        """
        return self._split_unstructured_text_sentence(text, chunk_size, chunk_overlap)
        
    def _split_unstructured_text_sentence(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int, int]]:
        """Split unstructured text using sentence-aware splitting"""
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
        
    def _split_unstructured_text_fixed_size(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int, int]]:
        """Split unstructured text into fixed-size chunks with overlap"""
        chunks = []
        
        # Simple fixed-size chunking with overlap
        for i in range(0, len(text), chunk_size - chunk_overlap):
            start_pos = i
            end_pos = min(i + chunk_size, len(text))
            
            # Try to break at sentence boundary
            if end_pos < len(text):
                # Look for sentence ending punctuation followed by space or newline
                for j in range(end_pos, max(start_pos, end_pos - 100), -1):
                    if j < len(text) and text[j] in '.!?' and (j + 1 == len(text) or text[j + 1].isspace()):
                        end_pos = j + 1
                        break
            
            chunk_text = text[start_pos:end_pos]
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append((chunk_text, start_pos, end_pos))
        
        return chunks
        
    def _split_unstructured_text_paragraph(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int, int]]:
        """Split unstructured text by paragraphs"""
        # Split text into paragraphs (delimiter: empty line)
        paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        start_pos = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            paragraph_length = len(paragraph)
            
            # If adding this paragraph would exceed chunk size and we already have content,
            # finalize the current chunk
            if current_chunk and (current_length + paragraph_length > chunk_size):
                chunk_text = "\n\n".join(current_chunk)
                end_pos = start_pos + len(chunk_text)
                chunks.append((chunk_text, start_pos, end_pos))
                
                # Start a new chunk with overlap
                if chunk_overlap > 0 and chunks:
                    # For paragraph-based splitting, overlap means including some paragraphs
                    # from the previous chunk
                    overlap_paragraphs = min(2, len(current_chunk))  # Take up to 2 paragraphs for overlap
                    overlap_text = "\n\n".join(current_chunk[-overlap_paragraphs:])
                    start_pos = end_pos - len(overlap_text)
                    current_chunk = current_chunk[-overlap_paragraphs:]
                    current_length = len(overlap_text)
                else:
                    start_pos = end_pos
                    current_chunk = []
                    current_length = 0
            
            # Add the paragraph to the current chunk
            current_chunk.append(paragraph)
            current_length += paragraph_length + 4  # +4 for the '\n\n'
        
        # Add the final chunk if there's anything left
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            end_pos = start_pos + len(chunk_text)
            chunks.append((chunk_text, start_pos, end_pos))
        
        return chunks
        
    def _split_unstructured_text_regex(self, text: str, chunk_size: int, chunk_overlap: int, regex_pattern: str) -> List[Tuple[str, int, int]]:
        """Split unstructured text using a custom regex pattern"""
        try:
            # Split text using the provided regex pattern
            regex = re.compile(regex_pattern)
            splits = []
            
            # Track positions while splitting
            last_end = 0
            for match in regex.finditer(text):
                if match.start() > last_end:
                    splits.append((text[last_end:match.start()], last_end, match.start()))
                last_end = match.end()
            
            # Add the final chunk
            if last_end < len(text):
                splits.append((text[last_end:], last_end, len(text)))
                
            # Process splits to ensure they don't exceed chunk_size
            chunks = []
            for split_text, start_pos, end_pos in splits:
                # If this split is too large, use fixed-size chunking on it
                if len(split_text) > chunk_size:
                    sub_chunks = self._split_unstructured_text_fixed_size(
                        split_text, chunk_size, chunk_overlap)
                    
                    # Adjust positions to be relative to the original text
                    for sub_text, sub_start, sub_end in sub_chunks:
                        chunks.append((sub_text, start_pos + sub_start, start_pos + sub_end))
                else:
                    chunks.append((split_text, start_pos, end_pos))
            
            return chunks
        except Exception as e:
            logger.warning(f"Error in regex chunking: {str(e)}")
            # Fallback to fixed-size chunking
            return self._split_unstructured_text_fixed_size(text, chunk_size, chunk_overlap)
    
    def _split_structured_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int, int]]:
        """
        Default split for structured text (fallback method)
        """
        return self._split_structured_text_row_based(text, 1, 10)
        
    def _split_structured_text_row_based(self, text: str, min_rows: int = 1, max_rows: int = 10) -> List[Tuple[str, int, int]]:
        """Split structured text by rows with header preservation"""
        import csv
        from io import StringIO
        
        chunks = []
        lines = text.splitlines()
        
        if not lines:
            return chunks
            
        # Try to detect CSV dialect
        try:
            sample = '\n'.join(lines[:min(10, len(lines))])
            dialect = csv.Sniffer().sniff(sample)
            has_header = csv.Sniffer().has_header(sample)
        except:
            # Default to comma separated if detection fails
            dialect = csv.excel
            has_header = True  # Assume first row is header for structured data
        
        # Extract header if present
        header = lines[0] if has_header else ""
        start_row = 1 if has_header else 0
        
        # Find position of first content row
        content_start_pos = text.find('\n') + 1 if has_header else 0
        
        # Track rows for chunking
        current_rows = []
        current_start = 0
        row_count = 0
        
        for i in range(start_row, len(lines)):
            line = lines[i]
            if not line.strip():  # Skip empty lines
                continue
                
            # Add this row
            current_rows.append(line)
            row_count += 1
            
            # If we've reached max_rows, create a chunk
            if row_count >= max_rows:
                # Create chunk with header
                if has_header:
                    chunk_lines = [header] + current_rows
                else:
                    chunk_lines = current_rows
                    
                chunk_text = '\n'.join(chunk_lines)
                current_end = current_start + len(chunk_text)
                chunks.append((chunk_text, current_start, current_end))
                
                # Reset for next chunk
                current_rows = []
                current_start = current_end
                row_count = 0
        
        # Add final chunk if there are remaining rows and we have at least min_rows
        if current_rows and row_count >= min_rows:
            if has_header:
                chunk_lines = [header] + current_rows
            else:
                chunk_lines = current_rows
                
            chunk_text = '\n'.join(chunk_lines)
            current_end = current_start + len(chunk_text)
            chunks.append((chunk_text, current_start, current_end))
        
        return chunks
    
    def _split_structured_text_multi_row(self, text: str, min_rows: int = 5, max_rows: int = 20) -> List[Tuple[str, int, int]]:
        """Split structured text by combining multiple rows based on sizes"""
        import csv
        from io import StringIO
        
        chunks = []
        lines = text.splitlines()
        
        if not lines:
            return chunks
            
        # Try to detect CSV dialect
        try:
            sample = '\n'.join(lines[:min(10, len(lines))])
            dialect = csv.Sniffer().sniff(sample)
            has_header = csv.Sniffer().has_header(sample)
        except:
            # Default to comma separated if detection fails
            dialect = csv.excel
            has_header = True  # Assume first row is header for structured data
        
        # Extract header if present
        header = lines[0] if has_header else ""
        start_row = 1 if has_header else 0
        
        # Track content size to make more balanced chunks
        row_lengths = []
        for i in range(start_row, len(lines)):
            line = lines[i].strip()
            if line:  # Skip empty lines
                row_lengths.append(len(line))
        
        # Calculate average row length to determine target chunk size
        avg_row_length = sum(row_lengths) / len(row_lengths) if row_lengths else 50
        target_row_count = max(min_rows, min(max_rows, int(1000 / avg_row_length)))
        
        # Now create chunks with this target size
        return self._split_structured_text_row_based(text, min_rows, target_row_count)
    
    def _split_semi_structured_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[Tuple[str, int, int]]:
        """Split semi-structured text (JSON, XML, Markdown, etc.) by semantic elements"""
        import json
        
        # Try to detect format based on content
        text = text.strip()
        is_json = (text.startswith('{') and text.endswith('}')) or (text.startswith('[') and text.endswith(']'))
        is_xml = text.startswith('<') and text.endswith('>')
        is_markdown = bool(re.search(r'^#+\s+', text, re.MULTILINE))
        
        chunks = []
        
        # JSON processing
        if is_json:
            try:
                data = json.loads(text)
                
                if isinstance(data, dict):
                    # Process JSON object by root keys
                    start_pos = 0
                    for key, value in data.items():
                        # Find the key in the original text
                        key_marker = f'"{key}"'
                        key_pos = text.find(key_marker, start_pos)
                        
                        if key_pos >= 0:
                            # Find the value end (next key or end of object)
                            if value is None:
                                value_str = "null"
                            elif isinstance(value, (int, float, bool)):
                                value_str = json.dumps(value)
                            else:
                                value_str = json.dumps(value)
                            
                            # Create a chunk with this key-value pair
                            chunk_text = f'{{{key_marker}: {value_str}}}'
                            end_pos = key_pos + len(chunk_text)
                            chunks.append((chunk_text, key_pos, end_pos))
                            start_pos = end_pos
                
                elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                    # Process JSON array of objects
                    buffer = []
                    buffer_length = 0
                    buffer_start = 0
                    
                    for i, item in enumerate(data):
                        item_json = json.dumps(item, ensure_ascii=False)
                        item_length = len(item_json)
                        
                        # If this item would make the buffer too big, flush it
                        if buffer and buffer_length + item_length > chunk_size:
                            chunk_text = f"[{','.join(buffer)}]"
                            chunks.append((chunk_text, buffer_start, buffer_start + len(chunk_text)))
                            buffer = [item_json]
                            buffer_length = item_length
                            buffer_start += len(chunk_text)
                        else:
                            buffer.append(item_json)
                            buffer_length += item_length + (2 if buffer else 0)  # Add comma and space when needed
                    
                    # Add remaining items in buffer
                    if buffer:
                        chunk_text = f"[{','.join(buffer)}]"
                        chunks.append((chunk_text, buffer_start, buffer_start + len(chunk_text)))
                
                else:
                    # Fall back to unstructured for other JSON types
                    return self._split_unstructured_text(text, chunk_size, chunk_overlap)
                
            except json.JSONDecodeError:
                # If JSON parsing fails, fall back to unstructured
                return self._split_unstructured_text(text, chunk_size, chunk_overlap)
        
        # XML/HTML processing
        elif is_xml:
            try:
                import xml.etree.ElementTree as ET
                
                # For XML, find major element boundaries
                depth = 0
                element_starts = []
                element_ends = []
                in_tag = False
                current_element_start = -1
                
                for i, char in enumerate(text):
                    if char == '<' and text[i+1:i+2] != '/':
                        in_tag = True
                        if depth == 0:
                            current_element_start = i
                        depth += 1
                    elif char == '<' and text[i+1:i+2] == '/':
                        in_tag = True
                    elif char == '>' and in_tag:
                        in_tag = False
                        if text[i-1:i] == '/':  # Self-closing tag
                            if depth > 0:
                                depth -= 1
                            if depth == 0 and current_element_start >= 0:
                                element_starts.append(current_element_start)
                                element_ends.append(i + 1)
                                current_element_start = -1
                        elif text[i-1:i-3:-1] == '/</':  # Closing tag
                            depth -= 1
                            if depth == 0 and current_element_start >= 0:
                                element_starts.append(current_element_start)
                                element_ends.append(i + 1)
                                current_element_start = -1
                
                # Create chunks from element boundaries
                for start, end in zip(element_starts, element_ends):
                    if end - start <= chunk_size:
                        chunks.append((text[start:end], start, end))
                    else:
                        # If element is too large, fall back to regular chunking for this element
                        element_chunks = self._split_unstructured_text(
                            text[start:end], chunk_size, chunk_overlap
                        )
                        for chunk_text, rel_start, rel_end in element_chunks:
                            abs_start = start + rel_start
                            abs_end = start + rel_end
                            chunks.append((chunk_text, abs_start, abs_end))
                
                # If no elements were processed, fall back to unstructured
                if not chunks:
                    return self._split_unstructured_text(text, chunk_size, chunk_overlap)
                
            except Exception as e:
                # If XML processing fails, fall back to unstructured
                return self._split_unstructured_text(text, chunk_size, chunk_overlap)
        
        # Markdown processing
        elif is_markdown:
            # Split by headings and content blocks
            lines = text.splitlines()
            heading_pattern = re.compile(r'^(#+)\s+(.+)$')
            
            current_section = []
            section_start = 0
            current_length = 0
            
            for i, line in enumerate(lines):
                line_with_newline = line + '\n'
                line_length = len(line_with_newline)
                
                # Check if this is a heading
                heading_match = heading_pattern.match(line)
                is_heading = bool(heading_match)
                heading_level = len(heading_match.group(1)) if heading_match else 0
                
                # If we hit a heading and already have content, finalize current section
                if is_heading and heading_level <= 2 and current_section and i > 0:
                    section_text = '\n'.join(current_section)
                    end_pos = section_start + len(section_text)
                    
                    if len(section_text) <= chunk_size:
                        chunks.append((section_text, section_start, end_pos))
                    else:
                        # If section is too large, split it further
                        sub_chunks = self._split_unstructured_text(
                            section_text, chunk_size, chunk_overlap
                        )
                        for chunk_text, rel_start, rel_end in sub_chunks:
                            abs_start = section_start + rel_start
                            abs_end = section_start + rel_end
                            chunks.append((chunk_text, abs_start, abs_end))
                    
                    # Start new section
                    current_section = [line]
                    section_start = end_pos
                    current_length = line_length
                else:
                    # Add line to current section
                    if not current_section:  # If this is the first line
                        section_start = 0
                    
                    current_section.append(line)
                    current_length += line_length
                    
                    # If section is getting too large, finalize it
                    if current_length >= chunk_size:
                        section_text = '\n'.join(current_section)
                        end_pos = section_start + len(section_text)
                        chunks.append((section_text, section_start, end_pos))
                        
                        # Start new section (empty)
                        current_section = []
                        section_start = end_pos
                        current_length = 0
            
            # Add final section if there's anything left
            if current_section:
                section_text = '\n'.join(current_section)
                end_pos = section_start + len(section_text)
                
                if len(section_text) <= chunk_size:
                    chunks.append((section_text, section_start, end_pos))
                else:
                    # If section is too large, split it further
                    sub_chunks = self._split_unstructured_text(
                        section_text, chunk_size, chunk_overlap
                    )
                    for chunk_text, rel_start, rel_end in sub_chunks:
                        abs_start = section_start + rel_start
                        abs_end = section_start + rel_end
                        chunks.append((chunk_text, abs_start, abs_end))
        
        else:
            # Fall back to unstructured for unknown formats
            return self._split_unstructured_text(text, chunk_size, chunk_overlap)
        
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
