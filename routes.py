import os
import json
import logging
from flask import render_template, request, jsonify, redirect, url_for, flash, send_file, current_app
from werkzeug.utils import secure_filename
from app import app, db
from models import Document, EmbeddingModel, Metadata, Chunk
from document_processor import DocumentProcessor
from embedding_models import EmbeddingModelManager
from vector_store import VectorStore

# Set up logging
logger = logging.getLogger(__name__)

# Initialize document processor outside app context as it doesn't need db access initially
document_processor = DocumentProcessor()

# Initialize components that require app context
embedding_manager = None
vector_store = None

# We'll initialize these variables inside application context right before they're needed
def get_embedding_manager():
    global embedding_manager
    if embedding_manager is None:
        embedding_manager = EmbeddingModelManager()
    return embedding_manager

def get_vector_store():
    global vector_store
    if vector_store is None:
        vector_store = VectorStore(get_embedding_manager())
    return vector_store

# Create a function to ensure a default model exists
def ensure_default_model():
    with app.app_context():
        if not EmbeddingModel.query.first():
            manager = get_embedding_manager()
            manager.add_model(
                name="Simple Vector Embedding",
                model_type="simple-embedder",
                model_path="default",
                dimension=384,
                is_active=True,
                is_default=True
            )

# Routes
@app.route('/')
def index():
    """Home page with recent documents and search box"""
    # Ensure we have a default model
    ensure_default_model()
    
    em = get_embedding_manager()
    
    recent_documents = Document.query.order_by(Document.updated_at.desc()).limit(5).all()
    active_models = em.get_active_models()
    default_model = em.get_default_model()
    
    return render_template(
        'index.html', 
        recent_documents=recent_documents,
        active_models=active_models,
        default_model=default_model
    )

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Document upload page and handling"""
    if request.method == 'POST':
        # Check if file was uploaded
        if 'document' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['document']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        # Read file content
        content = file.read().decode('utf-8')
        name = secure_filename(file.filename or 'unnamed_document.txt')
        mime_type = file.content_type or 'text/plain'
        
        # Extract metadata from form
        metadata = {}
        for key in request.form:
            if key.startswith('meta_key_'):
                idx = key.split('_')[-1]
                meta_key = request.form[f'meta_key_{idx}']
                meta_value = request.form[f'meta_value_{idx}']
                if meta_key and meta_value:
                    metadata[meta_key] = meta_value
        
        # Process the document
        try:
            document = document_processor.process_document(name, content, mime_type, metadata)
            
            # Generate embeddings for all active models
            em = get_embedding_manager()
            active_models = em.get_active_models()
            vs = get_vector_store()
            for model in active_models:
                vs.sync_document_embeddings(document.id, model.id)
            
            flash(f'Document "{name}" uploaded successfully', 'success')
            return redirect(url_for('document_list'))
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            flash(f'Error processing document: {str(e)}', 'error')
            return redirect(request.url)
    
    # GET request - show upload form
    return render_template('upload.html')

@app.route('/documents')
def document_list():
    """List all documents"""
    documents = Document.query.order_by(Document.updated_at.desc()).all()
    
    # Get metadata for each document
    documents_with_metadata = []
    for doc in documents:
        doc_metadata = Metadata.query.filter_by(document_id=doc.id).all()
        documents_with_metadata.append({
            'document': doc,
            'metadata': doc_metadata
        })
    
    return render_template('document_list.html', documents=documents_with_metadata)

@app.route('/documents/<document_id>')
def document_view(document_id):
    """View a document and its chunks"""
    document = Document.query.get_or_404(document_id)
    metadata = Metadata.query.filter_by(document_id=document_id).all()
    chunks = Chunk.query.filter_by(document_id=document_id).order_by(Chunk.chunk_index).all()
    
    return render_template(
        'document_view.html',
        document=document,
        metadata=metadata,
        chunks=chunks
    )

@app.route('/documents/<document_id>/edit', methods=['GET', 'POST'])
def document_edit(document_id):
    """Edit a document and its metadata"""
    document = Document.query.get_or_404(document_id)
    
    if request.method == 'POST':
        # Update document
        name = request.form.get('name')
        content = request.form.get('content')
        
        # Extract metadata from form
        metadata = {}
        for key in request.form:
            if key.startswith('meta_key_'):
                idx = key.split('_')[-1]
                meta_key = request.form[f'meta_key_{idx}']
                meta_value = request.form[f'meta_value_{idx}']
                if meta_key and meta_value:
                    metadata[meta_key] = meta_value
        
        # Update the document
        try:
            document_processor.update_document(document_id, name, content, metadata)
            
            # Re-generate embeddings
            vs = get_vector_store()
            vs.sync_document_embeddings(document_id)
            
            flash('Document updated successfully', 'success')
            return redirect(url_for('document_view', document_id=document_id))
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            flash(f'Error updating document: {str(e)}', 'error')
    
    # GET request - show edit form
    metadata = Metadata.query.filter_by(document_id=document_id).all()
    return render_template('document_edit.html', document=document, metadata=metadata)

@app.route('/documents/<document_id>/delete', methods=['POST'])
def document_delete(document_id):
    """Delete a document"""
    try:
        document_processor.delete_document(document_id)
        flash('Document deleted successfully', 'success')
        return redirect(url_for('document_list'))
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        flash(f'Error deleting document: {str(e)}', 'error')
        return redirect(url_for('document_view', document_id=document_id))

@app.route('/search')
def search_page():
    """Search page with form and results"""
    query = request.args.get('query', '')
    model_id = request.args.get('model_id', '')
    top_k = int(request.args.get('top_k', 5))
    min_score = float(request.args.get('min_score', 0.5))
    
    results = []
    em = get_embedding_manager()
    active_models = em.get_active_models()
    default_model = em.get_default_model()
    
    # If query provided, perform search
    if query:
        # Parse metadata filters
        metadata_filters = {}
        for key in request.args:
            if key.startswith('filter_key_'):
                idx = key.split('_')[-1]
                filter_key = request.args.get(f'filter_key_{idx}')
                filter_value = request.args.get(f'filter_value_{idx}')
                if filter_key and filter_value:
                    metadata_filters[filter_key] = filter_value
        
        try:
            model_id_to_use = model_id if model_id else (default_model.id if default_model else None)
            vs = get_vector_store()
            results = vs.search(
                query_text=query,
                model_id=model_id_to_use,
                top_k=top_k,
                min_score=min_score,
                metadata_filters=metadata_filters
            )
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            flash(f'Search error: {str(e)}', 'error')
    
    return render_template(
        'search.html',
        query=query,
        results=results,
        active_models=active_models,
        selected_model_id=model_id,
        default_model=default_model,
        top_k=top_k,
        min_score=min_score
    )

@app.route('/api/search', methods=['POST'])
def api_search():
    """API endpoint for search"""
    try:
        data = request.json or {}
        query = data.get('query', '')
        model_id = data.get('model_id')
        top_k = int(data.get('top_k', 5))
        min_score = float(data.get('min_score', 0.5))
        metadata_filters = data.get('metadata_filters', {})
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        vs = get_vector_store()
        results = vs.search(
            query_text=query,
            model_id=model_id,
            top_k=top_k,
            min_score=min_score,
            metadata_filters=metadata_filters
        )
        
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"API search error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/models')
def model_management():
    """Model management page"""
    models = EmbeddingModel.query.all()
    return render_template('model_management.html', models=models)

@app.route('/api/models', methods=['GET'])
def api_get_models():
    """API endpoint to get all models"""
    models = EmbeddingModel.query.all()
    model_list = []
    
    for model in models:
        model_list.append({
            'id': model.id,
            'name': model.name,
            'model_type': model.model_type,
            'model_path': model.model_path,
            'dimension': model.dimension,
            'is_active': model.is_active,
            'is_default': model.is_default,
            'config': model.config,
            'created_at': model.created_at.isoformat()
        })
    
    return jsonify({'models': model_list})

@app.route('/api/models', methods=['POST'])
def api_add_model():
    """API endpoint to add a new model"""
    try:
        data = request.json or {}
        
        required_fields = ['name', 'model_type', 'model_path', 'dimension']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Field {field} is required'}), 400
        
        em = get_embedding_manager()
        model = em.add_model(
            name=data['name'],
            model_type=data['model_type'],
            model_path=data['model_path'],
            dimension=int(data['dimension']),
            is_active=data.get('is_active', True),
            is_default=data.get('is_default', False),
            config=data.get('config')
        )
        
        return jsonify({
            'id': model.id,
            'name': model.name,
            'model_type': model.model_type,
            'model_path': model.model_path,
            'dimension': model.dimension,
            'is_active': model.is_active,
            'is_default': model.is_default,
            'config': model.config,
            'created_at': model.created_at.isoformat()
        }), 201
    except Exception as e:
        logger.error(f"Error adding model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>', methods=['PUT'])
def api_update_model(model_id):
    """API endpoint to update a model"""
    try:
        data = request.json or {}
        
        # Only include fields that are provided
        update_data = {}
        for field in ['name', 'model_type', 'model_path', 'dimension', 'is_active', 'is_default', 'config']:
            if field in data:
                update_data[field] = data[field]
        
        em = get_embedding_manager()
        model = em.update_model(model_id, **update_data)
        
        return jsonify({
            'id': model.id,
            'name': model.name,
            'model_type': model.model_type,
            'model_path': model.model_path,
            'dimension': model.dimension,
            'is_active': model.is_active,
            'is_default': model.is_default,
            'config': model.config,
            'created_at': model.created_at.isoformat()
        })
    except Exception as e:
        logger.error(f"Error updating model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>', methods=['DELETE'])
def api_delete_model(model_id):
    """API endpoint to delete a model"""
    try:
        em = get_embedding_manager()
        success = em.delete_model(model_id)
        
        if success:
            return '', 204
        else:
            return jsonify({'error': 'Failed to delete model'}), 500
    except Exception as e:
        logger.error(f"Error deleting model: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents/<document_id>/sync', methods=['POST'])
def api_sync_document(document_id):
    """API endpoint to sync document embeddings"""
    try:
        model_id = request.json.get('model_id') if request.json else None
        vs = get_vector_store()
        count = vs.sync_document_embeddings(document_id, model_id)
        return jsonify({'message': f'Synchronized {count} embeddings for document {document_id}'})
    except Exception as e:
        logger.error(f"Error syncing document: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chunking', methods=['GET'])
def api_get_chunking():
    """API endpoint to get chunking parameters"""
    return jsonify({
        'chunk_size': document_processor.chunk_size,
        'chunk_overlap': document_processor.chunk_overlap
    })

@app.route('/api/chunking', methods=['POST'])
def api_set_chunking():
    """API endpoint to set chunking parameters"""
    try:
        data = request.json
        
        chunk_size = int(data.get('chunk_size', 1000))
        chunk_overlap = int(data.get('chunk_overlap', 200))
        
        document_processor.set_chunking_parameters(chunk_size, chunk_overlap)
        
        return jsonify({
            'chunk_size': document_processor.chunk_size,
            'chunk_overlap': document_processor.chunk_overlap
        })
    except Exception as e:
        logger.error(f"Error setting chunking parameters: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Document Analysis and Processing Routes
@app.route('/document-schema')
def document_schema_page():
    """Document schema settings overview page"""
    # Get all documents with their metadata
    documents = []
    for doc in Document.query.all():
        doc_metadata = {}
        for meta in Metadata.query.filter_by(document_id=doc.id).all():
            doc_metadata[meta.key] = meta.value
        
        doc.metadata = doc_metadata
        documents.append(doc)
    
    # Calculate statistics
    stats = {
        'structured': 0,
        'semi_structured': 0,
        'unstructured': 0,
        'overrides': 0
    }
    
    for doc in documents:
        structure_type = doc.metadata.get('detected_structure', 'unstructured')
        stats[structure_type] += 1
        
        # Count manual overrides
        if 'structure_override' in doc.metadata and doc.metadata['structure_override'] == 'true':
            stats['overrides'] += 1
    
    return render_template('document_schema.html', documents=documents, stats=stats)

@app.route('/documents/<document_id>/analyze')
def document_analysis_page(document_id):
    """Detailed document analysis and configuration page"""
    from document_analyzer import DocumentAnalyzer
    
    # Analyze the document
    analyzer = DocumentAnalyzer(document_processor)
    analysis_result = analyzer.analyze_document(document_id)
    
    return render_template('document_analysis.html', **analysis_result)

@app.route('/api/documents/<document_id>/process', methods=['POST'])
def api_process_document(document_id):
    """API to process a document with specified configuration"""
    from document_analyzer import DocumentAnalyzer
    
    try:
        # Extract configuration from form data
        structure_type = request.form.get('structure_type', 'unstructured')
        language = request.form.get('language', 'en')
        
        # Get embedding and metadata fields
        embedding_fields = request.form.getlist('embedding_fields')
        metadata_fields = request.form.getlist('metadata_fields')
        store_full_content = request.form.get('store_full_content') == 'on'
        
        # Get custom metadata
        custom_meta_keys = request.form.getlist('custom_meta_key[]')
        custom_meta_values = request.form.getlist('custom_meta_value[]')
        custom_metadata = {}
        
        for i in range(min(len(custom_meta_keys), len(custom_meta_values))):
            key = custom_meta_keys[i].strip()
            value = custom_meta_values[i].strip()
            if key and value:
                custom_metadata[key] = value
        
        # Get chunking configuration based on structure type
        chunking_config = {}
        
        # Structured data chunking config
        if structure_type == 'structured':
            method = request.form.get('structured_strategy', 'row_based')
            min_rows = request.form.get('min_rows_per_chunk', 1)
            max_rows = request.form.get('max_rows_per_chunk', 10)
            
            chunking_config['structured'] = {
                'method': method,
                'min_rows_per_chunk': int(min_rows),
                'max_rows_per_chunk': int(max_rows)
            }
            
        # Semi-structured data chunking config
        elif structure_type == 'semi_structured':
            method = request.form.get('semi_structured_strategy', 'semantic_elements')
            element_path = request.form.get('element_path', '')
            preserve_hierarchy = request.form.get('preserve_hierarchy') == 'on'
            
            chunking_config['semi_structured'] = {
                'method': method,
                'element_path': element_path,
                'preserve_hierarchy': preserve_hierarchy
            }
            
        # Unstructured data chunking config
        else:
            method = request.form.get('unstructured_strategy', 'paragraph_based')
            chunk_size = request.form.get('chunk_size', 1000)
            chunk_overlap = request.form.get('chunk_overlap', 200)
            regex_pattern = request.form.get('regex_pattern', r'\n\s*\n|\r\n\s*\r\n')
            
            chunking_config['unstructured'] = {
                'method': method,
                'chunk_size': int(chunk_size),
                'chunk_overlap': int(chunk_overlap),
                'regex_pattern': regex_pattern
            }
        
        # Get embedding configuration
        embedding_model = request.form.get('embedding_model')
        embedding_dtype = request.form.get('embedding_dtype', 'float32')
        embedding_normalize = request.form.get('embedding_normalize', 'true') == 'true'
        max_tokens = request.form.get('max_tokens_per_chunk', 512)
        
        embedding_config = {
            'dtype': embedding_dtype,
            'normalize': embedding_normalize,
            'max_tokens_per_chunk': int(max_tokens)
        }
        
        # Create complete configuration
        config = {
            'detected_structure': structure_type,
            'detected_language': language,
            'embedding_fields': embedding_fields,
            'metadata_fields': metadata_fields,
            'store_full_content': store_full_content,
            'chunking_config': chunking_config,
            'embedding_config': embedding_config,
            'custom_metadata': custom_metadata,
            'selected_model_id': embedding_model
        }
        
        # Check if this is a save-only request
        save_only = request.form.get('save_only') == 'true'
        
        # Process the document or just save the configuration
        analyzer = DocumentAnalyzer(document_processor)
        
        if save_only:
            success = analyzer.save_processing_config(document_id, config)
            message = 'Configuration saved successfully'
        else:
            success = analyzer.process_document(document_id, config)
            message = 'Document processed successfully'
        
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'success': False, 'error': 'Failed to process document'}), 500
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/documents/<document_id>/preview', methods=['POST'])
def api_preview_document_processing(document_id):
    """API to generate preview chunks for document processing"""
    from document_analyzer import DocumentAnalyzer
    
    try:
        # Extract configuration from form data (similar to api_process_document)
        structure_type = request.form.get('structure_type', 'unstructured')
        
        # Get chunking configuration based on structure type
        chunking_config = {}
        
        # Structured data chunking config
        if structure_type == 'structured':
            method = request.form.get('structured_strategy', 'row_based')
            min_rows = request.form.get('min_rows_per_chunk', 1)
            max_rows = request.form.get('max_rows_per_chunk', 10)
            
            chunking_config = {
                'method': method,
                'min_rows_per_chunk': int(min_rows),
                'max_rows_per_chunk': int(max_rows)
            }
            
        # Semi-structured data chunking config
        elif structure_type == 'semi_structured':
            method = request.form.get('semi_structured_strategy', 'semantic_elements')
            element_path = request.form.get('element_path', '')
            preserve_hierarchy = request.form.get('preserve_hierarchy') == 'on'
            
            chunking_config = {
                'method': method,
                'element_path': element_path,
                'preserve_hierarchy': preserve_hierarchy
            }
            
        # Unstructured data chunking config
        else:
            method = request.form.get('unstructured_strategy', 'paragraph_based')
            chunk_size = request.form.get('chunk_size', 1000)
            chunk_overlap = request.form.get('chunk_overlap', 200)
            regex_pattern = request.form.get('regex_pattern', r'\n\s*\n|\r\n\s*\r\n')
            
            chunking_config = {
                'method': method,
                'chunk_size': int(chunk_size),
                'chunk_overlap': int(chunk_overlap),
                'regex_pattern': regex_pattern
            }
        
        # Create configuration for preview
        config = {
            'detected_structure': structure_type,
            'chunking_config': {structure_type: chunking_config}
        }
        
        # Generate preview chunks
        analyzer = DocumentAnalyzer(document_processor)
        preview_chunks = analyzer.generate_preview(document_id, config)
        
        return jsonify({
            'success': True, 
            'chunks': preview_chunks
        })
        
    except Exception as e:
        logger.error(f"Error generating preview: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/documents/<document_id>/structure', methods=['POST'])
def api_update_document_structure(document_id):
    """API endpoint to update document structure type and reprocess chunks"""
    try:
        data = request.json
        structure_type = data.get('structure_type', 'unstructured')
        
        if structure_type not in ['structured', 'semi_structured', 'unstructured']:
            return jsonify({'success': False, 'error': 'Invalid structure type'}), 400
        
        # Get the document
        document = Document.query.get_or_404(document_id)
        
        # Update metadata to indicate manual override
        metadata = {
            'detected_structure': structure_type,
            'detection_confidence': '100.0',  # Manual override has 100% confidence
            'structure_override': 'true'
        }
        
        # Reprocess document chunks using the new structure type
        document_processor.update_document(
            document_id=document_id,
            name=document.name,
            content=document.content,
            metadata=metadata
        )
        
        # Sync embeddings
        vs = get_vector_store()
        vs.sync_document_embeddings(document_id)
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Error updating document structure: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Admin Settings Routes
@app.route('/admin/settings')
def admin_settings_page():
    """Admin settings page"""
    from admin_settings import AdminSettings
    
    admin_settings = AdminSettings()
    
    return render_template(
        'admin_settings.html',
        faiss_settings=admin_settings.faiss_settings,
        embedding_settings=admin_settings.embedding_settings
    )

@app.route('/api/admin/settings', methods=['POST'])
def api_admin_settings():
    """API endpoint to save admin settings"""
    from admin_settings import AdminSettings
    
    try:
        data = request.json
        
        admin_settings = AdminSettings()
        
        # Update FAISS settings
        if 'faiss' in data:
            admin_settings.faiss_settings._update_from_dict(data['faiss'])
        
        # Update embedding settings
        if 'embedding' in data:
            admin_settings.embedding_settings._update_from_dict(data['embedding'])
        
        # Save all settings
        admin_settings.save()
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error saving admin settings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/settings/reset', methods=['POST'])
def api_admin_settings_reset():
    """API endpoint to reset admin settings to defaults"""
    try:
        # Remove saved settings from app config
        if 'FAISS_SETTINGS' in current_app.config:
            del current_app.config['FAISS_SETTINGS']
            
        if 'EMBEDDING_SETTINGS' in current_app.config:
            del current_app.config['EMBEDDING_SETTINGS']
        
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error resetting admin settings: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/admin/settings/rebuild-index', methods=['POST'])
def api_admin_rebuild_index():
    """API endpoint to rebuild the index with current settings"""
    from admin_settings import AdminSettings
    
    try:
        admin_settings = AdminSettings()
        vs = get_vector_store()
        
        success = admin_settings.rebuild_index(vs)
        
        return jsonify({'success': success})
    except Exception as e:
        logger.error(f"Error rebuilding index: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500
