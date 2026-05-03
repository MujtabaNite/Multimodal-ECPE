// MultiCauseNet - Professional Flask Frontend

// Example texts
const examples = {
    '1': 'I am so happy today! I got accepted to my dream university and my parents are so proud of me. This is the best news ever!',
    '2': 'I failed my final exam because I didn\'t study enough. I feel so disappointed in myself and worried about my future.',
    '3': 'The traffic made me late for the most important meeting of my career. I am so frustrated and angry at this situation!',
    '4': 'I heard strange noises in the basement. I\'m really scared that something might be wrong. My heart is racing.',
    '5': 'I can\'t believe it! My friends threw me a surprise party and got me the gift I\'ve been wanting for years. I never expected this!'
};

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    initializeTabs();
    initializeExamples();
    initializeFileUpload();
    initializeAnalyzeButtons();
});

// Tab switching
function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.getAttribute('data-tab');
            
            // Update buttons
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
            
            // Update content
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(`${tabName}-tab`).classList.add('active');
            
            // Hide results and errors
            hideResults();
            hideError();
        });
    });
}

// Example selector
function initializeExamples() {
    const select = document.getElementById('example-select');
    const textInput = document.getElementById('text-input');
    
    select.addEventListener('change', () => {
        const value = select.value;
        textInput.value = examples[value] || '';
    });
}

// File upload display
function initializeFileUpload() {
    const input = document.getElementById('audio-input');
    const fileName = document.getElementById('file-name');
    
    input.addEventListener('change', () => {
        if (input.files.length > 0) {
            fileName.textContent = input.files[0].name;
        } else {
            fileName.textContent = '';
        }
    });
}

// Analyze buttons
function initializeAnalyzeButtons() {
    document.getElementById('analyze-text-btn').addEventListener('click', analyzeText);
    document.getElementById('analyze-audio-btn').addEventListener('click', analyzeAudio);
}

// Analyze text
async function analyzeText() {
    const textInput = document.getElementById('text-input');
    const text = textInput.value.trim();
    
    if (!text) {
        showError('Please enter some text to analyze');
        return;
    }
    
    setLoading('text', true);
    hideError();
    hideResults();
    
    try {
        const response = await fetch('/api/analyze/text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayTextResults(data);
        } else {
            showError(data.error || 'Analysis failed');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    } finally {
        setLoading('text', false);
    }
}

// Analyze audio
async function analyzeAudio() {
    const input = document.getElementById('audio-input');
    
    if (!input.files || input.files.length === 0) {
        showError('Please upload an audio file');
        return;
    }
    
    setLoading('audio', true);
    hideError();
    hideResults();
    
    const formData = new FormData();
    formData.append('audio', input.files[0]);
    
    try {
        const response = await fetch('/api/analyze/audio', {
            method: 'POST',
            body: formData,
            // Don't set Content-Type header - let browser set it with boundary
        });
        
        // Check if response is OK
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: `Server error: ${response.status}` }));
            showError(errorData.error || `Request failed with status ${response.status}`);
            return;
        }
        
        const data = await response.json();
        
        if (data.success) {
            displayAudioResults(data);
        } else {
            showError(data.error || 'Analysis failed');
        }
    } catch (error) {
        console.error('Audio analysis error:', error);
        showError('Network error: ' + error.message + '. Please check console for details.');
    } finally {
        setLoading('audio', false);
    }
}

// Display text results
function displayTextResults(data) {
    const { emotions, causes } = data;
    
    // Hide transcript section
    document.getElementById('transcript-section').style.display = 'none';
    
    // Show all detected emotions (top 3)
    const metricsGrid = document.getElementById('metrics-grid');
    metricsGrid.innerHTML = emotions.map((emotion, index) => `
        <div class="metric-card">
            <div class="metric-label">
                ${index === 0 ? 'Primary Emotion' : `Emotion #${index + 1}`}
            </div>
            <div class="metric-value">${capitalizeFirst(emotion.emotion)}</div>
            <div class="metric-confidence">${(emotion.confidence * 100).toFixed(1)}% Confidence</div>
        </div>
    `).join('');
    
    // Show pairs
    displayPairs(causes);
    
    // Show results
    showResults();
}

// Display audio results
function displayAudioResults(data) {
    const { transcript, text_emotions, audio_emotions, analysis, causes } = data;
    
    // Show transcript
    document.getElementById('transcript-section').style.display = 'block';
    document.getElementById('transcript-text').textContent = transcript;
    
    // Show analysis explanation if available
    if (analysis) {
        const transcriptSection = document.getElementById('transcript-section');
        const analysisDiv = document.createElement('div');
        analysisDiv.className = 'result-card';
        analysisDiv.style.marginTop = '1rem';
        analysisDiv.style.borderLeft = '4px solid var(--secondary)';
        analysisDiv.innerHTML = `
            <h3 class="result-title">Analysis</h3>
            <p style="color: var(--gray-700); line-height: 1.6;">${analysis}</p>
        `;
        transcriptSection.appendChild(analysisDiv);
    }
    
    // Show emotions from both text and audio
    const metricsGrid = document.getElementById('metrics-grid');
    let emotionsHTML = '<div style="grid-column: 1/-1; margin-bottom: 0.5rem;"><h3 class="result-title">Text-Based Emotions (semantic content)</h3></div>';
    
    emotionsHTML += text_emotions.slice(0, 3).map((emotion, index) => `
        <div class="metric-card" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">
            <div class="metric-label">
                ${index === 0 ? 'Primary (Text)' : `Text #${index + 1}`}
            </div>
            <div class="metric-value">${capitalizeFirst(emotion.emotion)}</div>
            <div class="metric-confidence">${(emotion.confidence * 100).toFixed(1)}%</div>
        </div>
    `).join('');
    
    emotionsHTML += '<div style="grid-column: 1/-1; margin: 1rem 0 0.5rem;"><h3 class="result-title">Audio-Based Emotions (voice prosody)</h3></div>';
    
    emotionsHTML += audio_emotions.slice(0, 3).map((emotion, index) => `
        <div class="metric-card" style="background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);">
            <div class="metric-label">
                ${index === 0 ? 'Primary (Audio)' : `Audio #${index + 1}`}
            </div>
            <div class="metric-value">${capitalizeFirst(emotion.emotion)}</div>
            <div class="metric-confidence">${(emotion.confidence * 100).toFixed(1)}%</div>
        </div>
    `).join('');
    
    metricsGrid.innerHTML = emotionsHTML;
    
    // Show pairs
    displayPairs(causes);
    
    // Show results
    showResults();
}

// Display emotion-cause pairs
function displayPairs(pairs) {
    const container = document.getElementById('pairs-container');
    
    if (!pairs || pairs.length === 0) {
        container.innerHTML = `
            <div class="result-card">
                <p style="color: var(--gray-600);">No clear emotion-cause pairs detected. Try a different input with more explicit causal language.</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = pairs.map(pair => {
        const confidenceLevel = pair.confidence >= 0.75 ? 'high' : 
                               pair.confidence >= 0.5 ? 'medium' : 'low';
        const confidenceText = confidenceLevel.charAt(0).toUpperCase() + confidenceLevel.slice(1);
        
        return `
            <div class="pair-card">
                <div class="pair-header">
                    <span class="pair-id">Pair #${pair.pair_id}</span>
                    <span class="pair-confidence confidence-${confidenceLevel}">
                        ${confidenceText} (${(pair.confidence * 100).toFixed(0)}%)
                    </span>
                </div>
                <div class="pair-emotion">
                    <strong>Emotion:</strong> ${capitalizeFirst(pair.emotion)}
                </div>
                <div class="pair-cause">
                    "${pair.cause}"
                </div>
                <div class="pair-evidence">
                    Evidence: ${pair.evidence}
                </div>
            </div>
        `;
    }).join('');
}

// UI helpers
function setLoading(type, loading) {
    const btn = document.getElementById(`analyze-${type}-btn`);
    const text = btn.querySelector('.btn-text');
    const loader = btn.querySelector('.btn-loader');
    
    if (loading) {
        text.style.display = 'none';
        loader.style.display = 'inline-block';
        btn.disabled = true;
    } else {
        text.style.display = 'inline';
        loader.style.display = 'none';
        btn.disabled = false;
    }
}

function showResults() {
    const results = document.getElementById('results');
    results.style.display = 'block';
    results.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideResults() {
    document.getElementById('results').style.display = 'none';
}

function showError(message) {
    const error = document.getElementById('error');
    error.textContent = message;
    error.style.display = 'block';
    error.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function hideError() {
    document.getElementById('error').style.display = 'none';
}

function capitalizeFirst(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}
