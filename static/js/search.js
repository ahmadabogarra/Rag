/**
 * JavaScript for the search functionality
 */

document.addEventListener('DOMContentLoaded', function() {
    // Search form handling
    const searchForm = document.getElementById('search-form');
    if (searchForm) {
        // Handle form submission - already handled by form action
        
        // Save search preferences to localStorage
        searchForm.addEventListener('submit', function() {
            const modelId = document.getElementById('model-select').value;
            const topK = document.getElementById('top-k').value;
            const minScore = document.getElementById('min-score').value;
            
            localStorage.setItem('search_model_id', modelId);
            localStorage.setItem('search_top_k', topK);
            localStorage.setItem('search_min_score', minScore);
        });
        
        // Load search preferences from localStorage
        const savedModelId = localStorage.getItem('search_model_id');
        const savedTopK = localStorage.getItem('search_top_k');
        const savedMinScore = localStorage.getItem('search_min_score');
        
        const modelSelect = document.getElementById('model-select');
        const topKInput = document.getElementById('top-k');
        const minScoreInput = document.getElementById('min-score');
        
        if (savedModelId && modelSelect && !modelSelect.value) {
            modelSelect.value = savedModelId;
        }
        
        if (savedTopK && topKInput && !topKInput.value) {
            topKInput.value = savedTopK;
        }
        
        if (savedMinScore && minScoreInput && !minScoreInput.value) {
            minScoreInput.value = savedMinScore;
        }
    }

    // Advanced search options toggle
    const advancedToggle = document.getElementById('toggle-advanced');
    const advancedOptions = document.getElementById('advanced-options');
    
    if (advancedToggle && advancedOptions) {
        advancedToggle.addEventListener('click', function() {
            if (advancedOptions.style.display === 'none') {
                advancedOptions.style.display = 'block';
                advancedToggle.textContent = 'Hide Advanced Options';
            } else {
                advancedOptions.style.display = 'none';
                advancedToggle.textContent = 'Show Advanced Options';
            }
        });
    }

    // Search results highlighting
    const highlightSearchResults = function() {
        const searchQuery = new URLSearchParams(window.location.search).get('query');
        if (!searchQuery) return;
        
        const terms = searchQuery.split(' ').filter(term => term.length > 2);
        const contentElements = document.querySelectorAll('.list-group-item .p-2');
        
        contentElements.forEach(element => {
            let content = element.innerHTML;
            
            // Highlight each term
            terms.forEach(term => {
                const regex = new RegExp('(' + term + ')', 'gi');
                content = content.replace(regex, '<mark>$1</mark>');
            });
            
            element.innerHTML = content;
        });
    };
    
    // Call the highlighting function
    highlightSearchResults();

    // Metadata filter handling - Dynamic filter fields
    const addFilterBtn = document.getElementById('add-filter');
    const filtersContainer = document.getElementById('filters-container');
    
    if (addFilterBtn && filtersContainer) {
        // Add filter handling is already in the template
        
        // Handle filter dropdown population
        const populateFilterKeys = async function() {
            try {
                // Get common metadata keys
                const response = await fetch('/api/metadata/keys');
                if (!response.ok) return;
                
                const data = await response.json();
                if (!data.keys || !data.keys.length) return;
                
                // Create datalist for keys
                const datalist = document.createElement('datalist');
                datalist.id = 'metadata-keys';
                
                data.keys.forEach(key => {
                    const option = document.createElement('option');
                    option.value = key;
                    datalist.appendChild(option);
                });
                
                document.body.appendChild(datalist);
                
                // Connect all key inputs to datalist
                document.querySelectorAll('input[name^="filter_key_"]').forEach(input => {
                    input.setAttribute('list', 'metadata-keys');
                });
            } catch (error) {
                console.error('Error fetching metadata keys:', error);
            }
        };
        
        // This API endpoint doesn't exist yet but can be added in the future
        // populateFilterKeys();
    }
});
