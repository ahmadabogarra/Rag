
function updateFieldPreview() {
    // تحديث حقول التضمين المحددة
    const embeddingFields = Array.from(document.querySelectorAll('input[name="embedding_fields"]:checked'))
        .map(input => input.value);
    
    const embeddingList = document.getElementById('selectedEmbeddingFields');
    embeddingList.innerHTML = embeddingFields.length ? 
        embeddingFields.map(field => `<li><span class="badge bg-primary">${field}</span></li>`).join('') :
        '<li class="text-muted">لم يتم تحديد أي حقول للتضمين</li>';
    
    // تحديث البيانات الوصفية المحددة
    const metadataFields = Array.from(document.querySelectorAll('input[name="metadata_fields"]:checked'))
        .map(input => input.value);
    
    const metadataList = document.getElementById('selectedMetadataFields');
    metadataList.innerHTML = metadataFields.length ?
        metadataFields.map(field => `<li><span class="badge bg-info">${field}</span></li>`).join('') :
        '<li class="text-muted">لم يتم تحديد أي حقول للبيانات الوصفية</li>';
}

// تحديث العرض عند تحميل الصفحة
document.addEventListener('DOMContentLoaded', function() {
    updateFieldPreview();
});

/**
 * Document Animations - Playful loading animations with document-themed spinners
 */

class DocumentAnimations {
    constructor() {
        this.animationTypes = [
            'document-flip',
            'document-shuffle',
            'document-search',
            'loading-spinner'
        ];

        this.init();
    }

    init() {
        // Initialize animations when the page loads
        this.setupAnimationContainers();
        
        // Listen for AJAX events to show/hide loading animations
        this.setupAjaxListeners();
        
        // Listen for form submissions to show loading animations
        this.setupFormListeners();
    }

    setupAnimationContainers() {
        // Find all animation containers
        const containers = document.querySelectorAll('.loading-container[data-animation-type]');
        
        containers.forEach(container => {
            const animationType = container.getAttribute('data-animation-type');
            if (this.animationTypes.includes(animationType)) {
                // Create the specific animation
                this.createAnimation(container, animationType);
            } else {
                // Default to loading spinner
                this.createAnimation(container, 'loading-spinner');
            }
        });
    }

    setupAjaxListeners() {
        // Show animations during AJAX requests
        document.addEventListener('ajax:loading', (event) => {
            const targetId = event.detail?.targetId;
            const animationType = event.detail?.animationType || 'loading-spinner';
            
            if (targetId) {
                const container = document.getElementById(targetId);
                if (container) {
                    this.showAnimation(container, animationType);
                }
            }
        });

        document.addEventListener('ajax:complete', (event) => {
            const targetId = event.detail?.targetId;
            
            if (targetId) {
                const container = document.getElementById(targetId);
                if (container) {
                    this.hideAnimation(container);
                }
            }
        });
    }

    setupFormListeners() {
        // Show animations during form submissions
        const forms = document.querySelectorAll('form[data-show-loading]');
        
        forms.forEach(form => {
            form.addEventListener('submit', (event) => {
                const targetId = form.getAttribute('data-loading-target');
                const animationType = form.getAttribute('data-animation-type') || 'loading-spinner';
                
                if (targetId) {
                    const container = document.getElementById(targetId);
                    if (container) {
                        this.showAnimation(container, animationType);
                    }
                }
            });
        });
    }

    // Create animation based on type
    createAnimation(container, type) {
        // Clear previous content
        container.innerHTML = '';
        
        switch (type) {
            case 'document-flip':
                this.createDocumentFlip(container);
                break;
            case 'document-shuffle':
                this.createDocumentShuffle(container);
                break;
            case 'document-search':
                this.createDocumentSearch(container);
                break;
            case 'loading-spinner':
            default:
                this.createLoadingSpinner(container);
                break;
        }
    }

    // Create a document flip animation
    createDocumentFlip(container) {
        const docFlip = document.createElement('div');
        docFlip.className = 'document-flip';
        
        // Add pages
        for (let i = 0; i < 5; i++) {
            const page = document.createElement('div');
            page.className = 'page';
            docFlip.appendChild(page);
        }
        
        container.appendChild(docFlip);
    }

    // Create a document shuffle animation
    createDocumentShuffle(container) {
        const docShuffle = document.createElement('div');
        docShuffle.className = 'document-shuffle';
        
        // Add documents
        for (let i = 0; i < 3; i++) {
            const doc = document.createElement('div');
            doc.className = 'doc';
            docShuffle.appendChild(doc);
        }
        
        container.appendChild(docShuffle);
    }

    // Create a document search animation
    createDocumentSearch(container) {
        const docSearch = document.createElement('div');
        docSearch.className = 'document-search';
        
        // Add document
        const doc = document.createElement('div');
        doc.className = 'doc';
        docSearch.appendChild(doc);
        
        // Add magnifier
        const magnifier = document.createElement('div');
        magnifier.className = 'magnifier';
        docSearch.appendChild(magnifier);
        
        container.appendChild(docSearch);
    }

    // Create a simple loading spinner
    createLoadingSpinner(container) {
        const spinner = document.createElement('div');
        spinner.className = 'loading-spinner';
        container.appendChild(spinner);
    }

    // Show animation in container
    showAnimation(container, type) {
        // First ensure the container has the right class
        container.classList.add('loading-container');
        
        // Set the animation type
        container.setAttribute('data-animation-type', type);
        
        // Show the container if it was hidden
        container.style.display = 'flex';
        
        // Create the animation
        this.createAnimation(container, type);
    }

    // Hide animation
    hideAnimation(container) {
        // Hide the container
        container.style.display = 'none';
        
        // Clear the content
        container.innerHTML = '';
    }
}

// Initialize the animations when the DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    window.documentAnimations = new DocumentAnimations();

    // Add a global helper method to trigger animations manually
    window.showDocumentLoader = function(containerId, animationType) {
        if (window.documentAnimations) {
            const container = document.getElementById(containerId);
            if (container) {
                window.documentAnimations.showAnimation(container, animationType);
            }
        }
    };

    window.hideDocumentLoader = function(containerId) {
        if (window.documentAnimations) {
            const container = document.getElementById(containerId);
            if (container) {
                window.documentAnimations.hideAnimation(container);
            }
        }
    };
});