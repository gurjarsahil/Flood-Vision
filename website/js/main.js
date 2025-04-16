/*
 * FloodSense AI - Advanced Flood Prediction System
 * Main JavaScript File
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initNavigation();
    initDemoSection();
    initContactForm();
    initAnimations();
});

/**
 * Navigation functionality
 */
function initNavigation() {
    // Mobile menu toggle
    const menuToggle = document.querySelector('.mobile-menu-toggle');
    const navMenu = document.querySelector('.nav-menu');
    
    if (menuToggle && navMenu) {
        menuToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            menuToggle.classList.toggle('active');
        });
    }

    // Smooth scrolling for navigation links
    const navLinks = document.querySelectorAll('a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Close mobile menu if open
            if (navMenu && navMenu.classList.contains('active')) {
                navMenu.classList.remove('active');
                if (menuToggle) menuToggle.classList.remove('active');
            }
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80, // Adjust for header height
                    behavior: 'smooth'
                });
            }
        });
    });

    // Highlight active nav item on scroll
    window.addEventListener('scroll', highlightNavOnScroll);
}

function highlightNavOnScroll() {
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-menu a');
    
    let currentSection = '';
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop - 100;
        const sectionHeight = section.offsetHeight;
        
        if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
            currentSection = '#' + section.getAttribute('id');
        }
    });
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === currentSection) {
            link.classList.add('active');
        }
    });
}

/**
 * Demo section functionality
 */
function initDemoSection() {
    // Initialize tab navigation
    initDemoTabs();
    
    // Initialize range sliders
    initRangeSliders();
    
    // Form submission handling
    const demoForm = document.getElementById('demo-form');
    const resultContainer = document.getElementById('prediction-result');
    const loadingIndicator = document.getElementById('loading-indicator');
    
    if (demoForm) {
        demoForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading indicator
            if (loadingIndicator) loadingIndicator.style.display = 'block';
            if (resultContainer) resultContainer.style.display = 'none';
            
            // Get form data
            const formData = new FormData(demoForm);
            
            // Simulate API call with timeout
            setTimeout(() => {
                // Process prediction (in a real app, this would be an API call)
                const mockPrediction = simulateFloodPrediction(formData);
                
                // Hide loading indicator
                if (loadingIndicator) loadingIndicator.style.display = 'none';
                
                // Display result
                if (resultContainer) {
                    resultContainer.innerHTML = generateResultHTML(mockPrediction);
                    resultContainer.style.display = 'block';
                }
            }, 1500);
        });
    }
    
    // Handle image upload form
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        const fileInput = document.getElementById('demo-file');
        const filePreview = document.getElementById('file-preview');
        
        // Show preview of uploaded image
        if (fileInput && filePreview) {
            fileInput.addEventListener('change', function() {
                filePreview.innerHTML = '';
                if (this.files && this.files[0]) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        const img = document.createElement('img');
                        img.src = e.target.result;
                        img.classList.add('preview-image');
                        filePreview.appendChild(img);
                    }
                    reader.readAsDataURL(this.files[0]);
                }
            });
        }
        
        // Handle form submission
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            // In a real app, you would upload the image to your backend
            alert('Image analysis feature will be available in the full version.');
        });
    }
}

/**
 * Initialize the demo section tabs
 */
function initDemoTabs() {
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    tabBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            // Remove active class from all buttons and panes
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));
            
            // Add active class to current button
            this.classList.add('active');
            
            // Show the target pane
            const targetId = this.getAttribute('data-target');
            const targetPane = document.querySelector(targetId);
            if (targetPane) {
                targetPane.classList.add('active');
            }
        });
    });
}

/**
 * Initialize range sliders with value display
 */
function initRangeSliders() {
    const rangeSliders = document.querySelectorAll('.range-slider');
    
    rangeSliders.forEach(slider => {
        const valueDisplay = slider.nextElementSibling;
        
        // Set initial value
        updateRangeValue(slider, valueDisplay);
        
        // Update on change
        slider.addEventListener('input', function() {
            updateRangeValue(this, valueDisplay);
        });
    });
}

/**
 * Update the displayed value for a range slider
 */
function updateRangeValue(slider, display) {
    if (!display) return;
    
    const value = slider.value;
    const id = slider.id;
    
    // Format display based on slider type
    if (id === 'rainfall') {
        display.textContent = `${value} mm`;
    } else if (id === 'river-level' || id === 'soil-saturation') {
        display.textContent = `${value}%`;
    } else {
        display.textContent = value;
    }
}

/**
 * Simulate a flood prediction based on input data
 * In a real app, this would be an API call to a backend service
 */
function simulateFloodPrediction(formData) {
    // Get values from form data
    const rainfall = parseFloat(formData.get('rainfall'));
    const riverLevel = parseFloat(formData.get('river-level'));
    const soilSaturation = parseFloat(formData.get('soil-saturation'));
    
    // Calculate risk score (simplified algorithm)
    // Values are weighted: rainfall (40%), river level (35%), soil saturation (25%)
    const riskScore = (rainfall * 0.4) + (riverLevel * 0.35) + (soilSaturation * 0.25);
    
    // Determine risk level
    let riskLevel;
    let recommendations = [];
    
    if (riskScore < 30) {
        riskLevel = 'Low';
        recommendations = [
            'Continue to monitor weather forecasts',
            'No immediate action required'
        ];
    } else if (riskScore < 60) {
        riskLevel = 'Moderate';
        recommendations = [
            'Be prepared for possible flooding',
            'Check emergency supplies',
            'Stay informed about weather updates'
        ];
    } else if (riskScore < 80) {
        riskLevel = 'High';
        recommendations = [
            'Consider moving valuables to higher ground',
            'Prepare for possible evacuation',
            'Follow local authority instructions',
            'Avoid flood-prone areas'
        ];
    } else {
        riskLevel = 'Severe';
        recommendations = [
            'Evacuate if advised by local authorities',
            'Move to higher ground immediately',
            'Avoid walking or driving through flood waters',
            'Follow emergency broadcast instructions'
        ];
    }
    
    return {
        inputs: {
            rainfall,
            riverLevel,
            soilSaturation
        },
        results: {
            riskScore: Math.min(Math.round(riskScore), 100),
            riskLevel,
            recommendations
        }
    };
}

/**
 * Generate HTML for displaying prediction results
 */
function generateResultHTML(prediction) {
    const { inputs, results } = prediction;
    const { riskScore, riskLevel, recommendations } = results;
    
    // Determine color class based on risk level
    let colorClass;
    switch (riskLevel) {
        case 'Low':
            colorClass = 'text-success';
            break;
        case 'Moderate':
            colorClass = 'text-warning';
            break;
        case 'High':
        case 'Severe':
            colorClass = 'text-danger';
            break;
        default:
            colorClass = '';
    }
    
    // Build recommendations list
    const recommendationsList = recommendations.map(rec => `<li>${rec}</li>`).join('');
    
    // Build the result HTML
    return `
        <div class="prediction-summary">
            <h4>Flood Risk Assessment</h4>
            <div class="risk-indicator">
                <div class="risk-level ${colorClass}">
                    <span class="risk-value">${riskScore}%</span>
                    <span class="risk-label">${riskLevel} Risk</span>
                </div>
            </div>
            
            <div class="prediction-details">
                <h5>Input Parameters:</h5>
                <ul>
                    <li>Recent Rainfall: ${inputs.rainfall} mm</li>
                    <li>River Level: ${inputs.riverLevel}%</li>
                    <li>Soil Saturation: ${inputs.soilSaturation}%</li>
                </ul>
                
                <h5>Recommendations:</h5>
                <ul class="recommendations">
                    ${recommendationsList}
                </ul>
            </div>
        </div>
    `;
}

/**
 * Contact form functionality
 */
function initContactForm() {
    const contactForm = document.getElementById('contact-form');
    const formMessage = document.getElementById('form-message');
    
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // In a real implementation, you would send this data to your backend
            // For demo purposes, we'll just show a success message
            
            if (formMessage) {
                formMessage.innerHTML = '<div class="success-message">Thank you for your message! We will get back to you soon.</div>';
                formMessage.style.display = 'block';
            }
            
            // Reset form
            contactForm.reset();
            
            // Hide message after a few seconds
            setTimeout(() => {
                if (formMessage) formMessage.style.display = 'none';
            }, 5000);
        });
    }
}

/**
 * Animation and UI enhancements
 */
function initAnimations() {
    // Animate elements when they come into view
    const animatedElements = document.querySelectorAll('.animate-on-scroll');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animated');
            }
        });
    }, { threshold: 0.1 });
    
    animatedElements.forEach(element => {
        observer.observe(element);
    });
    
    // Parallax effect for hero section
    const heroSection = document.querySelector('.hero');
    if (heroSection) {
        window.addEventListener('scroll', () => {
            const scrollPosition = window.scrollY;
            if (scrollPosition < window.innerHeight) {
                heroSection.style.backgroundPositionY = `${scrollPosition * 0.4}px`;
            }
        });
    }
}

// Utility functions
function scrollToSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        window.scrollTo({
            top: section.offsetTop - 80,
            behavior: 'smooth'
        });
    }
} 