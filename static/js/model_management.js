/**
 * JavaScript for embedding model management
 */

document.addEventListener('DOMContentLoaded', function() {
    // Get model data for editing
    const loadModelData = async function(modelId) {
        try {
            const response = await fetch(`/api/models`);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            const models = data.models;
            
            // Find the model with the given ID
            const model = models.find(m => m.id === modelId);
            if (!model) {
                throw new Error(`Model with ID ${modelId} not found`);
            }
            
            return model;
        } catch (error) {
            console.error('Error loading model data:', error);
            window.showError(`Failed to load model data: ${error.message}`);
            return null;
        }
    };
    
    // Add new model
    const addModelForm = document.getElementById('add-model-form');
    const saveNewModelBtn = document.getElementById('save-new-model');
    const addModelModal = document.getElementById('addModelModal');
    
    if (addModelForm && saveNewModelBtn) {
        saveNewModelBtn.addEventListener('click', async function() {
            // Validate form
            const name = document.getElementById('model-name').value.trim();
            const modelType = document.getElementById('model-type').value;
            const modelPath = document.getElementById('model-path').value.trim();
            const dimension = parseInt(document.getElementById('model-dimension').value);
            const isActive = document.getElementById('model-active').checked;
            const isDefault = document.getElementById('model-default').checked;
            
            if (!name) {
                window.showError('Model name is required');
                return;
            }
            
            if (!modelPath) {
                window.showError('Model path/identifier is required');
                return;
            }
            
            if (isNaN(dimension) || dimension <= 0) {
                window.showError('Vector dimension must be a positive number');
                return;
            }
            
            // Disable button and show loading state
            saveNewModelBtn.disabled = true;
            const originalText = saveNewModelBtn.innerHTML;
            saveNewModelBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Saving...';
            
            try {
                const response = await fetch('/api/models', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        name: name,
                        model_type: modelType,
                        model_path: modelPath,
                        dimension: dimension,
                        is_active: isActive,
                        is_default: isDefault
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                
                // Hide modal
                const modal = bootstrap.Modal.getInstance(addModelModal);
                modal.hide();
                
                // Show success message
                window.showSuccess(`Model "${name}" added successfully`);
                
                // Refresh page to show new model
                setTimeout(() => {
                    location.reload();
                }, 1000);
            } catch (error) {
                console.error('Error adding model:', error);
                window.showError(`Failed to add model: ${error.message}`);
            } finally {
                // Re-enable button and restore text
                saveNewModelBtn.disabled = false;
                saveNewModelBtn.innerHTML = originalText;
            }
        });
    }
    
    // Edit model
    const editModelButtons = document.querySelectorAll('.edit-model');
    const editModelForm = document.getElementById('edit-model-form');
    const updateModelBtn = document.getElementById('update-model');
    const editModelModal = document.getElementById('editModelModal');
    
    if (editModelButtons.length && editModelForm && updateModelBtn) {
        editModelButtons.forEach(button => {
            button.addEventListener('click', async function() {
                const modelId = this.dataset.modelId;
                document.getElementById('edit-model-id').value = modelId;
                
                // Disable button and show loading state
                button.disabled = true;
                const originalText = button.innerHTML;
                button.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
                
                try {
                    const model = await loadModelData(modelId);
                    if (!model) return;
                    
                    // Fill form with model data
                    document.getElementById('edit-model-name').value = model.name;
                    document.getElementById('edit-model-path').value = model.model_path;
                    document.getElementById('edit-model-dimension').value = model.dimension;
                    document.getElementById('edit-model-active').checked = model.is_active;
                    document.getElementById('edit-model-default').checked = model.is_default;
                    
                    // Disable default checkbox if already default
                    if (model.is_default) {
                        document.getElementById('edit-model-default').disabled = true;
                    } else {
                        document.getElementById('edit-model-default').disabled = false;
                    }
                } catch (error) {
                    console.error('Error loading model for edit:', error);
                    window.showError(`Failed to load model data: ${error.message}`);
                } finally {
                    // Re-enable button and restore text
                    button.disabled = false;
                    button.innerHTML = originalText;
                }
            });
        });
        
        updateModelBtn.addEventListener('click', async function() {
            const modelId = document.getElementById('edit-model-id').value;
            
            // Validate form
            const name = document.getElementById('edit-model-name').value.trim();
            const modelPath = document.getElementById('edit-model-path').value.trim();
            const dimension = parseInt(document.getElementById('edit-model-dimension').value);
            const isActive = document.getElementById('edit-model-active').checked;
            const isDefault = document.getElementById('edit-model-default').checked;
            
            if (!name) {
                window.showError('Model name is required');
                return;
            }
            
            if (!modelPath) {
                window.showError('Model path/identifier is required');
                return;
            }
            
            if (isNaN(dimension) || dimension <= 0) {
                window.showError('Vector dimension must be a positive number');
                return;
            }
            
            // Disable button and show loading state
            updateModelBtn.disabled = true;
            const originalText = updateModelBtn.innerHTML;
            updateModelBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Saving...';
            
            try {
                const response = await fetch(`/api/models/${modelId}`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        name: name,
                        model_path: modelPath,
                        dimension: dimension,
                        is_active: isActive,
                        is_default: isDefault
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                
                // Hide modal
                const modal = bootstrap.Modal.getInstance(editModelModal);
                modal.hide();
                
                // Show success message
                window.showSuccess(`Model "${name}" updated successfully`);
                
                // Refresh page to show updated model
                setTimeout(() => {
                    location.reload();
                }, 1000);
            } catch (error) {
                console.error('Error updating model:', error);
                window.showError(`Failed to update model: ${error.message}`);
            } finally {
                // Re-enable button and restore text
                updateModelBtn.disabled = false;
                updateModelBtn.innerHTML = originalText;
            }
        });
    }
    
    // Delete model
    const deleteModelButtons = document.querySelectorAll('.delete-model');
    const confirmDeleteBtn = document.getElementById('confirm-delete-model');
    const deleteModelModal = new bootstrap.Modal(document.getElementById('deleteModelModal'));
    
    if (deleteModelButtons.length && confirmDeleteBtn) {
        let modelToDelete = null;
        
        deleteModelButtons.forEach(button => {
            button.addEventListener('click', function() {
                modelToDelete = {
                    id: this.dataset.modelId,
                    name: this.dataset.modelName
                };
                
                // Update modal text
                document.getElementById('delete-model-name').textContent = modelToDelete.name;
                
                // Show modal
                deleteModelModal.show();
            });
        });
        
        confirmDeleteBtn.addEventListener('click', async function() {
            if (!modelToDelete) return;
            
            // Disable button and show loading state
            confirmDeleteBtn.disabled = true;
            const originalText = confirmDeleteBtn.innerHTML;
            confirmDeleteBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Deleting...';
            
            try {
                const response = await fetch(`/api/models/${modelToDelete.id}`, {
                    method: 'DELETE'
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                // Hide modal
                deleteModelModal.hide();
                
                // Show success message
                window.showSuccess(`Model "${modelToDelete.name}" deleted successfully`);
                
                // Refresh page to update model list
                setTimeout(() => {
                    location.reload();
                }, 1000);
            } catch (error) {
                console.error('Error deleting model:', error);
                window.showError(`Failed to delete model: ${error.message}`);
            } finally {
                // Re-enable button and restore text
                confirmDeleteBtn.disabled = false;
                confirmDeleteBtn.innerHTML = originalText;
            }
        });
    }
    
    // Toggle model status (active/inactive)
    const toggleStatusButtons = document.querySelectorAll('.toggle-model-status');
    
    if (toggleStatusButtons.length) {
        toggleStatusButtons.forEach(button => {
            button.addEventListener('click', async function() {
                const modelId = this.dataset.modelId;
                const currentStatus = parseInt(this.dataset.currentStatus);
                const newStatus = !currentStatus;
                
                // Get model name from table row
                const row = this.closest('tr');
                const modelName = row.cells[0].textContent;
                
                // Disable button and show loading state
                button.disabled = true;
                const originalText = button.innerHTML;
                button.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
                
                try {
                    const response = await fetch(`/api/models/${modelId}`, {
                        method: 'PUT',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            is_active: newStatus
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! Status: ${response.status}`);
                    }
                    
                    // Show success message
                    window.showSuccess(`Model "${modelName}" ${newStatus ? 'activated' : 'deactivated'} successfully`);
                    
                    // Refresh page to update status
                    setTimeout(() => {
                        location.reload();
                    }, 1000);
                } catch (error) {
                    console.error('Error toggling model status:', error);
                    window.showError(`Failed to update model status: ${error.message}`);
                } finally {
                    // Re-enable button and restore text
                    button.disabled = false;
                    button.innerHTML = originalText;
                }
            });
        });
    }
    
    // Set default model
    const setDefaultButtons = document.querySelectorAll('.set-default-model');
    
    if (setDefaultButtons.length) {
        setDefaultButtons.forEach(button => {
            button.addEventListener('click', async function() {
                const modelId = this.dataset.modelId;
                
                // Get model name from table row
                const row = this.closest('tr');
                const modelName = row.cells[0].textContent;
                
                // Disable button and show loading state
                button.disabled = true;
                const originalText = button.innerHTML;
                button.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
                
                try {
                    const response = await fetch(`/api/models/${modelId}`, {
                        method: 'PUT',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            is_default: true
                        })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! Status: ${response.status}`);
                    }
                    
                    // Show success message
                    window.showSuccess(`Model "${modelName}" set as default successfully`);
                    
                    // Refresh page to update status
                    setTimeout(() => {
                        location.reload();
                    }, 1000);
                } catch (error) {
                    console.error('Error setting default model:', error);
                    window.showError(`Failed to set default model: ${error.message}`);
                } finally {
                    // Re-enable button and restore text
                    button.disabled = false;
                    button.innerHTML = originalText;
                }
            });
        });
    }
});
