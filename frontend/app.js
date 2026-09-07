document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const uploadZone = document.getElementById('upload-zone');
    const audioInput = document.getElementById('audio-input');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const removeFileBtn = document.getElementById('remove-file');
    const audioPreview = document.getElementById('audio-preview');
    const audioPlayer = document.getElementById('audio-player');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultCard = document.getElementById('result-card');
    
    // Result Elements
    const ringProgress = document.getElementById('ring-progress');
    const aiPercent = document.getElementById('ai-percent');
    const verdictTitle = document.getElementById('verdict-title');
    const verdictSubtitle = document.getElementById('verdict-subtitle');
    const explanationBox = document.getElementById('explanation-box');

    let currentFile = null;

    // -- File Handling --
    function handleFile(file) {
        if (!file || !file.type.startsWith('audio/')) {
            alert('Please select a valid audio file.');
            return;
        }

        currentFile = file;
        
        // Update UI
        uploadPlaceholder.classList.add('hidden');
        fileInfo.classList.remove('hidden');
        audioPreview.classList.remove('hidden');
        
        fileName.textContent = file.name;
        fileSize.textContent = (file.size / 1024).toFixed(2) + ' KB';
        
        const url = URL.createObjectURL(file);
        audioPlayer.src = url;
        
        analyzeBtn.disabled = false;
        resultCard.classList.add('hidden');
    }

    function clearFile() {
        currentFile = null;
        audioInput.value = '';
        
        if (audioPlayer.src) {
            URL.revokeObjectURL(audioPlayer.src);
            audioPlayer.src = '';
        }
        
        uploadPlaceholder.classList.remove('hidden');
        fileInfo.classList.add('hidden');
        audioPreview.classList.add('hidden');
        
        analyzeBtn.disabled = true;
        resultCard.classList.add('hidden');
    }

    // -- Event Listeners --
    uploadPlaceholder.addEventListener('click', () => audioInput.click());
    
    audioInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    removeFileBtn.addEventListener('click', clearFile);

    // Drag and Drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = '#00e5ff';
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.style.borderColor = '';
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor = '';
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    // -- Analysis --
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // Reset UI to loading
        resultCard.classList.remove('hidden');
        verdictTitle.textContent = "Analyzing...";
        verdictSubtitle.textContent = "";
        aiPercent.textContent = "0%";
        ringProgress.style.strokeDashoffset = 339.292;
        ringProgress.classList.remove('stroke-red', 'stroke-cyan', 'stroke-amber');
        verdictTitle.classList.remove('text-red', 'text-cyan', 'text-amber');
        explanationBox.innerHTML = '';
        analyzeBtn.disabled = true;
        
        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/api/detect', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Server error');
            }

            displayResult(data);
        } catch (error) {
            verdictTitle.textContent = "Error";
            verdictSubtitle.textContent = error.message;
            verdictTitle.classList.add('text-red');
        } finally {
            analyzeBtn.disabled = false;
        }
    });

    function displayResult(data) {
        // Set Ring progress
        const circumference = 339.292; // 2 * pi * 54
        const aiProb = data.ai_probability || 0;
        const offset = circumference - (aiProb / 100) * circumference;
        ringProgress.style.strokeDashoffset = offset;
        
        aiPercent.textContent = Math.round(aiProb) + '%';
        
        // Color mapping - handle both backend response formats
        let colorClass = '';
        let title = '';
        const pred = (data.prediction || '').toLowerCase();
        
        if (pred === 'ai_generated' || pred === 'likely_ai_generated') {
            colorClass = 'red';
            title = 'AI GENERATED';
        } else if (pred === 'real' || pred === 'likely_human') {
            colorClass = 'cyan';
            title = 'REAL / HUMAN';
        } else {
            colorClass = 'amber';
            title = 'UNCERTAIN';
        }
        
        ringProgress.classList.add(`stroke-${colorClass}`);
        verdictTitle.classList.add(`text-${colorClass}`);
        verdictTitle.textContent = title;
        
        // risk_level may not exist in all backends
        const riskLevel = data.risk_level || (aiProb > 65 ? 'HIGH' : aiProb < 35 ? 'LOW' : 'MEDIUM');
        verdictSubtitle.textContent = `${data.confidence}% confidence | ${riskLevel} risk`;
        
        // Explanation - handle both 'explanation' and 'details' fields
        let expHtml = '';
        const explanations = data.explanation || data.details || [];
        if (Array.isArray(explanations)) {
            explanations.forEach(p => expHtml += `<p>${p}</p>`);
        }
        explanationBox.innerHTML = expHtml;
    }
    
    // Status check
    fetch('/api/status').then(r => r.json()).then(data => {
        if (!data.ready) {
            document.querySelector('.status-text').textContent = 'Warning: Model not trained';
            document.querySelector('.status-dot').style.backgroundColor = '#ffb700';
            document.querySelector('.status-dot').style.boxShadow = '0 0 8px #ffb700';
        }
    }).catch(e => console.error("Could not fetch status:", e));
});
