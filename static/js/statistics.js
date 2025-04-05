// static/js/statistics.js
document.addEventListener('DOMContentLoaded', function() {
    // Create the usage chart
    const ctx = document.getElementById('usageChart').getContext('2d');
    
    // Define colors explicitly instead of using CSS variables
    const safeColor = '#28a745';  // Green color for safe content
    const hateColor = '#dc3545';  // Red color for hate speech
    
    // Chart configuration
    const usageChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Safe Content', 'Hate Speech'],
            datasets: [{
                data: [76, 24],
                backgroundColor: [
                    safeColor,
                    hateColor
                ],
                borderColor: [
                    safeColor,
                    hateColor
                ],
                borderWidth: 1,
                hoverOffset: 15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.label + ': ' + context.raw + '%';
                        }
                    }
                }
            },
            animation: {
                animateScale: true,
                animateRotate: true,
                duration: 2000,
                easing: 'easeOutQuart'
            },
            cutout: '65%'
        }
    });
    
    // Create animation for stats numbers
    const statValues = document.querySelectorAll('.stat-value');
    statValues.forEach(statValue => {
        const finalValue = statValue.textContent;
        statValue.textContent = '0%';
        
        // Animate the counter
        setTimeout(() => {
            animateCounter(statValue, 0, parseInt(finalValue), 1500);
        }, 500);
    });
    
    // Tag cloud animation
    const tags = document.querySelectorAll('.tag');
    tags.forEach((tag, index) => {
        // Random delay for each tag
        const delay = 500 + (index * 100);
        setTimeout(() => {
            tag.style.opacity = '1';
            tag.style.transform = 'translateY(0)';
        }, delay);
    });
});

// Function to animate counting
function animateCounter(element, start, end, duration) {
    const range = end - start;
    const increment = range / (duration / 16); // 16ms is approx one frame
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            clearInterval(timer);
            current = end;
        }
        element.textContent = Math.round(current) + '%';
    }, 16);
}

// Add hover effects for interactive statistics
document.querySelectorAll('.stat-item').forEach(item => {
    item.addEventListener('mouseenter', function() {
        this.querySelector('.stat-value').style.color = '#1a8fe3';  // Hover color
    });
    
    item.addEventListener('mouseleave', function() {
        this.querySelector('.stat-value').style.color = '#1da1f2';  // Primary color
    });
});