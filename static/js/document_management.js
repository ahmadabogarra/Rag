/**
 * JavaScript for document management functionality
 */

document.addEventListener('DOMContentLoaded', function() {
    // Document upload form handling
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const fileInput = document.getElementById('document');
            if (fileInput && fileInput.files.length === 0) {
                e.preventDefault();
                window.showError('Please select a file to upload');
                return false;
            }
            
            // Validate metadata
            const metadataKeys = document.querySelectorAll('input[name^="meta_key_"]');
            const metadataValues = document.querySelectorAll('input[name^="meta_value_"]');
            
            for (let i = 0; i < metadataKeys.length; i++) {
                const key = metadataKeys[i].value.trim();
                const value = metadataValues[i].value.trim();
                
                if (key && !value) {
                    e.preventDefault();
                    window.showError('Please provide a value for all metadata keys');
                    metadataValues[i].focus();
                    return false;
                }
                
                if (!key && value) {
                    e.preventDefault();
                    window.showError('Please provide a key for all metadata values');
                    metadataKeys[i].focus();
                    return false;
                }
            }
            
            // Show loading indicator
            const submitButton = uploadForm.querySelector('button[type="submit"]');
            submitButton.disabled = true;
            submitButton.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Uploading...';
        });
    }
    
    // Document edit form handling
    const editForm = document.getElementById('edit-form');
    if (editForm) {
        editForm.addEventListener('submit', function(e) {
            // Validate metadata
            const metadataKeys = document.querySelectorAll('input[name^="meta_key_"]');
            const metadataValues = document.querySelectorAll('input[name^="meta_value_"]');
            
            for (let i = 0; i < metadataKeys.length; i++) {
                const key = metadataKeys[i].value.trim();
                const value = metadataValues[i].value.trim();
                
                if (key && !value) {
                    e.preventDefault();
                    window.showError('Please provide a value for all metadata keys');
                    metadataValues[i].focus();
                    return false;
                }
                
                if (!key && value) {
                    e.preventDefault();
                    window.showError('Please provide a key for all metadata values');
                    metadataKeys[i].focus();
                    return false;
                }
            }
            
            // Show loading indicator
            const submitButton = editForm.querySelector('button[type="submit"]');
            submitButton.disabled = true;
            submitButton.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Saving...';
        });
    }
    
    // Document list filtering
    const documentSearch = document.getElementById('document-search');
    if (documentSearch) {
        documentSearch.addEventListener('keyup', function() {
            const searchText = this.value.toLowerCase();
            const documentRows = document.querySelectorAll('.document-row');
            
            documentRows.forEach(row => {
                const text = row.textContent.toLowerCase();
                if (text.includes(searchText)) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        });
    }
    
    // Document chunking preview
    const chunkSizeInput = document.getElementById('chunk-size');
    const chunkOverlapInput = document.getElementById('chunk-overlap');
    const previewButton = document.getElementById('preview-chunks');
    const previewContainer = document.getElementById('chunk-preview');
    
    if (previewButton && previewContainer) {
        previewButton.addEventListener('click', function() {
            const fileInput = document.getElementById('document');
            if (!fileInput || fileInput.files.length === 0) {
                window.showError('Please select a file to preview chunks');
                return;
            }
            
            const chunkSize = parseInt(chunkSizeInput.value);
            const chunkOverlap = parseInt(chunkOverlapInput.value);
            
            if (isNaN(chunkSize) || chunkSize < 100) {
                window.showError('Chunk size must be at least 100 characters');
                chunkSizeInput.focus();
                return;
            }
            
            if (isNaN(chunkOverlap) || chunkOverlap < 0 || chunkOverlap >= chunkSize) {
                window.showError('Chunk overlap must be between 0 and chunk size');
                chunkOverlapInput.focus();
                return;
            }
            
            const file = fileInput.files[0];
            const reader = new FileReader();
            
            reader.onload = function(e) {
                const content = e.target.result;
                
                // Simple chunking preview (actual chunking is done on the server)
                const chunks = [];
                let start = 0;
                
                while (start < content.length) {
                    const end = Math.min(start + chunkSize, content.length);
                    chunks.push({
                        content: content.substring(start, end),
                        start: start,
                        end: end
                    });
                    start = end - chunkOverlap;
                }
                
                // Display chunks preview
                previewContainer.innerHTML = '';
                
                if (chunks.length > 0) {
                    const previewElement = document.createElement('div');
                    previewElement.innerHTML = `<p>Estimated ${chunks.length} chunks will be created:</p>`;
                    
                    const chunkList = document.createElement('div');
                    chunkList.className = 'accordion';
                    chunkList.id = 'chunksPreviewAccordion';
                    
                    chunks.slice(0, 5).forEach((chunk, index) => {
                        const chunkItem = document.createElement('div');
                        chunkItem.className = 'accordion-item';
                        chunkItem.innerHTML = `
                            <h2 class="accordion-header" id="preview-heading-${index}">
                                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" 
                                        data-bs-target="#preview-collapse-${index}" aria-expanded="false" 
                                        aria-controls="preview-collapse-${index}">
                                    Chunk #${index + 1} (Pos: ${chunk.start}-${chunk.end})
                                </button>
                            </h2>
                            <div id="preview-collapse-${index}" class="accordion-collapse collapse" 
                                 aria-labelledby="preview-heading-${index}" data-bs-parent="#chunksPreviewAccordion">
                                <div class="accordion-body">
                                    <pre class="chunk-content p-2 bg-dark text-light rounded">${chunk.content}</pre>
                                </div>
                            </div>
                        `;
                        chunkList.appendChild(chunkItem);
                    });
                    
                    if (chunks.length > 5) {
                        const moreChunks = document.createElement('p');
                        moreChunks.className = 'text-muted mt-2';
                        moreChunks.textContent = `... and ${chunks.length - 5} more chunks`;
                        chunkList.appendChild(moreChunks);
                    }
                    
                    previewElement.appendChild(chunkList);
                    previewContainer.appendChild(previewElement);
                } else {
                    previewContainer.innerHTML = '<p class="text-muted">No chunks would be created for this content.</p>';
                }
            };
            
            reader.readAsText(file);
        });
    }
    
    // Chunk visualization in document view
    const contentTab = document.getElementById('content-tab');
    const chunksTab = document.getElementById('chunks-tab');
    
    if (contentTab && chunksTab) {
        // Highlight current chunk when selected
        const highlightChunk = function(chunkId) {
            // Remove previous highlights
            document.querySelectorAll('.chunk-highlight').forEach(el => {
                el.classList.remove('chunk-highlight');
            });
            
            const chunk = document.querySelector(`#${chunkId}`);
            if (chunk) {
                const chunkData = chunk.dataset;
                const startPos = parseInt(chunkData.startPos);
                const endPos = parseInt(chunkData.endPos);
                
                // Apply highlight in full content view
                const contentElement = document.querySelector('#content pre');
                if (contentElement) {
                    const content = contentElement.textContent;
                    const beforeChunk = content.substring(0, startPos);
                    const chunkContent = content.substring(startPos, endPos);
                    const afterChunk = content.substring(endPos);
                    
                    contentElement.innerHTML = `
                        ${beforeChunk}<span class="chunk-highlight">${chunkContent}</span>${afterChunk}
                    `;
                    
                    // Switch to content tab
                    document.querySelector('#content-tab').click();
                    
                    // Scroll to highlight
                    const highlight = contentElement.querySelector('.chunk-highlight');
                    if (highlight) {
                        highlight.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                }
            }
        };
        
        // Add click handlers for chunk headers
        document.querySelectorAll('.accordion-button').forEach(button => {
            button.addEventListener('click', function() {
                const collapseId = this.getAttribute('data-bs-target').substring(1);
                const chunkId = collapseId.replace('collapse', 'chunk');
                
                // Get associated chunk data
                const chunkElement = document.getElementById(chunkId);
                if (chunkElement) {
                    // Store selected chunk in session storage
                    sessionStorage.setItem('selected_chunk', chunkId);
                }
            });
        });
        
        // Restore selected chunk on page load
        const selectedChunkId = sessionStorage.getItem('selected_chunk');
        if (selectedChunkId) {
            const chunkElement = document.getElementById(selectedChunkId);
            if (chunkElement) {
                const collapseId = selectedChunkId.replace('chunk', 'collapse');
                const collapseElement = document.getElementById(collapseId);
                
                if (collapseElement) {
                    // Open the accordion item
                    const accordion = new bootstrap.Collapse(collapseElement, {
                        toggle: true
                    });
                    
                    // Scroll to it
                    chunkElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
            }
        }
    }
});
