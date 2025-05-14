/**
 * Main JavaScript file for Document Management System
 */

// Initialize loading animations
document.addEventListener('DOMContentLoaded', function() {
    // Initialize loading containers
    const loadingContainers = document.querySelectorAll('.loading-container');
    loadingContainers.forEach(container => {
        container.style.display = 'none';
    });

    // Initialize Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize Bootstrap popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
    
    // Helper function to show errors
    window.showError = function(message, container) {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-danger alert-dismissible fade show';
        alertDiv.innerHTML = `
            <i class="fas fa-exclamation-circle me-1"></i>${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        if (container) {
            container.prepend(alertDiv);
        } else {
            document.querySelector('.container').prepend(alertDiv);
        }
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alertDiv);
            bsAlert.close();
        }, 5000);
    };
    
    // Helper function to show success messages
    window.showSuccess = function(message, container) {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'alert alert-success alert-dismissible fade show';
        alertDiv.innerHTML = `
            <i class="fas fa-check-circle me-1"></i>${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        if (container) {
            container.prepend(alertDiv);
        } else {
            document.querySelector('.container').prepend(alertDiv);
        }
        
        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alertDiv);
            bsAlert.close();
        }, 5000);
    };
});

// Toggle animation function
function toggleAnimation(containerId, show) {
    const container = document.getElementById(containerId);
    if (container) {
        container.style.display = show ? 'flex' : 'none';
    }
}

// Global error handler
window.addEventListener('error', function(event) {
    console.error('Error:', event.error);
});