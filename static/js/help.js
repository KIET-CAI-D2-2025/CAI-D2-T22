// static/js/help.js
document.addEventListener('DOMContentLoaded', function() {
    // FAQ accordion functionality
    const faqItems = document.querySelectorAll('.faq-item');
    
    faqItems.forEach(item => {
        const question = item.querySelector('.faq-question');
        
        question.addEventListener('click', () => {
            // Toggle active class on the clicked item
            item.classList.toggle('active');
            
            // Close other FAQs
            faqItems.forEach(otherItem => {
                if (otherItem !== item) {
                    otherItem.classList.remove('active');
                }
            });
        });
    });
    
    // Contact form validation and submission
    const contactForm = document.querySelector('.contact-form');
    
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const nameInput = document.getElementById('name');
            const emailInput = document.getElementById('email');
            const messageInput = document.getElementById('message');
            const submitBtn = document.querySelector('.submit-btn');
            
            // Simple validation
            let isValid = true;
            
            if (!nameInput.value.trim()) {
                highlightError(nameInput);
                isValid = false;
            } else {
                removeError(nameInput);
            }
            
            if (!emailInput.value.trim() || !isValidEmail(emailInput.value)) {
                highlightError(emailInput);
                isValid = false;
            } else {
                removeError(emailInput);
            }
            
            if (!messageInput.value.trim()) {
                highlightError(messageInput);
                isValid = false;
            } else {
                removeError(messageInput);
            }
            
            if (isValid) {
                // Simulate form submission
                submitBtn.innerHTML = '<span>Sending...</span><i class="fas fa-spinner fa-spin"></i>';
                submitBtn.disabled = true;
                
                // Simulate API call
                setTimeout(() => {
                    submitBtn.innerHTML = '<span>Sent!</span><i class="fas fa-check"></i>';
                    submitBtn.style.backgroundColor = 'var(--success-color)';
                    
                    // Reset form after delay
                    setTimeout(() => {
                        contactForm.reset();
                        submitBtn.innerHTML = '<span>Submit</span><i class="fas fa-paper-plane"></i>';
                        submitBtn.style.backgroundColor = 'var(--primary-color)';
                        submitBtn.disabled = false;
                    }, 2000);
                }, 1500);
            }
        });
    }
    
    // Add visual feedback when stepping through the how-to-use steps
    const steps = document.querySelectorAll('.step');
    
    steps.forEach(step => {
        step.addEventListener('mouseenter', function() {
            steps.forEach(s => s.style.opacity = '0.6');
            this.style.opacity = '1';
        });
        
        step.addEventListener('mouseleave', function() {
            steps.forEach(s => s.style.opacity = '1');
        });
    });
});

// Helper functions for form validation
function isValidEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

function highlightError(inputElement) {
    inputElement.style.borderColor = 'var(--error-color)';
    inputElement.style.boxShadow = '0 0 0 3px rgba(224, 36, 94, 0.2)';
}

function removeError(inputElement) {
    inputElement.style.borderColor = 'var(--border-color)';
    inputElement.style.boxShadow = 'none';
}