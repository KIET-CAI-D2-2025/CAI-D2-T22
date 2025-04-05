// static/js/script.js
document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('prediction-form');
    const resultCard = document.getElementById('result-card');
    const loader = document.getElementById('loader');
    const analyzedText = document.getElementById('analyzed-text');
    const gaugeFill = document.getElementById('gauge-fill');
    const gaugePercentage = document.getElementById('gauge-percentage');
    const resultLabel = document.getElementById('result-label');
    const alertContainer = document.getElementById('alert-container');
    const aggressiveWordsContainer = document.getElementById('aggressive-words-container');
    const aggressiveWordsList = document.getElementById('aggressive-words-list');
    const aggressiveCount = document.getElementById('aggressive-count');
    const severityFill = document.getElementById('severity-fill');
    const severityPercentage = document.getElementById('severity-percentage');
    const mobileMenuToggle = document.getElementById('mobile-menu-toggle');
    const navLinks = document.getElementById('nav-links');

    // Navigation scroll effect
    let lastScrollTop = 0;
    const navbar = document.querySelector('.navbar');
    
    window.addEventListener('scroll', function() {
        let scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        // Add shadow effect when scrolling down
        if (scrollTop > 10) {
            navbar.style.boxShadow = '0 4px 10px rgba(0, 0, 0, 0.1)';
            navbar.style.height = '50px';
        } else {
            navbar.style.boxShadow = '0 2px 10px rgba(0, 0, 0, 0.1)';
            navbar.style.height = 'var(--nav-height)';
        }
        
        // Hide navbar when scrolling down, show when scrolling up
        if (scrollTop > 200) { // Only apply this behavior after scrolling a bit
            if (scrollTop > lastScrollTop) {
                // Scrolling down
                navbar.style.top = '-60px';
            } else {
                // Scrolling up
                navbar.style.top = '0';
            }
        } else {
            navbar.style.top = '0';
        }
        
        lastScrollTop = scrollTop;
    });
    
    // Close dropdown when clicking elsewhere
    document.addEventListener('click', function(event) {
        // Only if the dropdown is already open
        const navLinks = document.getElementById('nav-links');
        if (navLinks && navLinks.classList.contains('active')) {
            // If click is outside the navbar
            if (!event.target.closest('.navbar')) {
                // Close the dropdown
                navLinks.classList.remove('active');
                
                // Change icon back to bars
                const menuToggle = document.getElementById('mobile-menu-toggle');
                if (menuToggle) {
                    const icon = menuToggle.querySelector('i');
                    if (icon) {
                        icon.classList.remove('fa-times');
                        icon.classList.add('fa-bars');
                    }
                }
            }
        }
    });

    if (mobileMenuToggle && navLinks) {
        mobileMenuToggle.addEventListener('click', function() {
            navLinks.classList.toggle('active');
            
            // Change icon based on menu state
            const icon = this.querySelector('i');
            if (navLinks.classList.contains('active')) {
                icon.classList.remove('fa-bars');
                icon.classList.add('fa-times');
            } else {
                icon.classList.remove('fa-times');
                icon.classList.add('fa-bars');
            }
        });
        
        // Close menu when clicking outside
        document.addEventListener('click', function(event) {
            if (!event.target.closest('.nav-container') && navLinks.classList.contains('active')) {
                navLinks.classList.remove('active');
                const icon = mobileMenuToggle.querySelector('i');
                icon.classList.remove('fa-times');
                icon.classList.add('fa-bars');
            }
        });
        
        // Close menu when clicking on a link
        const navLinkItems = navLinks.querySelectorAll('a');
        navLinkItems.forEach(link => {
            link.addEventListener('click', function() {
                if (window.innerWidth <= 768) {
                    navLinks.classList.remove('active');
                    const icon = mobileMenuToggle.querySelector('i');
                    icon.classList.remove('fa-times');
                    icon.classList.add('fa-bars');
                }
            });
        });
        
        // Handle window resize
        window.addEventListener('resize', function() {
            if (window.innerWidth > 768 && navLinks.classList.contains('active')) {
                navLinks.classList.remove('active');
                const icon = mobileMenuToggle.querySelector('i');
                icon.classList.remove('fa-times');
                icon.classList.add('fa-bars');
            }
        });
    }
    
    // Add active class to current page nav link
    const currentPage = window.location.pathname;
    const navLinkItems = document.querySelectorAll('.nav-links a');
    
    navLinkItems.forEach(link => {
        const linkPath = link.getAttribute('href');
        if (currentPage === linkPath || 
            (currentPage === '/' && linkPath.includes('home'))) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
    
    if (!predictionForm) return;
    
    predictionForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Hide result card and show loader
        resultCard.classList.add('hidden');
        alertContainer.classList.add('hidden');
        loader.classList.remove('hidden');
        
        const formData = new FormData(predictionForm);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            
            const result = await response.json();
            
            if (result.error) {
                alert(result.error);
                return;
            }
            
            // Update result card with prediction
            analyzedText.innerHTML = result.highlighted_text; // Use highlighted text from server
            
            // Update gauge
            const percentage = Math.round(result.probability * 100);
            gaugeFill.style.height = `${percentage}%`;
            gaugePercentage.textContent = `${percentage}%`;
            
            // Update result label
            if (result.is_hate_speech) {
                resultLabel.textContent = 'Hate Speech Detected';
                resultLabel.className = 'result-label hate-speech';
                
                // Show alert for hate speech
                alertContainer.classList.remove('hidden');
                alertContainer.classList.add('visible');
            } else {
                resultLabel.textContent = 'Not Hate Speech';
                resultLabel.className = 'result-label not-hate-speech';
                
                // Hide alert for non-hate speech
                alertContainer.classList.add('hidden');
                alertContainer.classList.remove('visible');
            }
            
            // Update aggressive words list
            aggressiveWordsList.innerHTML = '';
            if (result.aggressive_words && result.aggressive_words.length > 0) {
                aggressiveCount.textContent = result.aggressive_count;
                
                // Create list items for each aggressive word
                result.aggressive_words.forEach(word => {
                    const li = document.createElement('li');
                    li.textContent = word;
                    aggressiveWordsList.appendChild(li);
                });
                
                // Update severity meter
                severityFill.style.width = `${result.severity}%`;
                severityPercentage.textContent = `${result.severity}%`;
                
                // Set different colors based on severity
                if (result.severity < 30) {
                    severityFill.style.backgroundColor = 'var(--success-color)';
                } else if (result.severity < 70) {
                    severityFill.style.backgroundColor = 'var(--warning-color)';
                } else {
                    severityFill.style.backgroundColor = 'var(--error-color)';
                }
                
                // Show aggressive words container
                aggressiveWordsContainer.classList.remove('hidden');
            } else {
                // Hide aggressive words container if no aggressive words found
                aggressiveWordsContainer.classList.add('hidden');
            }
            
            // Hide loader and show result card
            loader.classList.add('hidden');
            resultCard.classList.remove('hidden');
            resultCard.classList.add('visible');
            
            // Add smooth scrolling to result
            resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
            
            // Add shake animation if hate speech detected
            if (result.is_hate_speech) {
                resultCard.classList.add('shake');
                setTimeout(() => {
                    resultCard.classList.remove('shake');
                }, 1000);
            }
            
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing your request.');
            loader.classList.add('hidden');
        }
    });
    
    // Add keydown event to form to submit on Enter key
    document.getElementById('tweet-input').addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            e.preventDefault();
            document.getElementById('analyze-btn').click();
        }
    });
});

// Add smooth scrolling to all anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});