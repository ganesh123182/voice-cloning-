document.addEventListener('DOMContentLoaded', () => {
    // --- State ---
    let token = localStorage.getItem('token');
    let isLoginMode = true;
    let enrollFile = null;
    let monitorFile = null;

    // --- DOM Elements ---
    const authView = document.getElementById('auth-view');
    const dashboardView = document.getElementById('dashboard-view');
    const userStatus = document.getElementById('user-status');
    const userEmailDisplay = document.getElementById('user-email-display');
    
    // Auth Form
    const authForm = document.getElementById('auth-form');
    const authTitle = document.getElementById('auth-title');
    const authSubmitBtn = document.getElementById('auth-submit-btn');
    const authToggleBtn = document.getElementById('auth-toggle-btn');
    const authToggleText = document.getElementById('auth-toggle-text');
    const authError = document.getElementById('auth-error');
    
    // Tabs
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Enrollment
    const enrollInput = document.getElementById('enroll-audio-input');
    const enrollZone = document.getElementById('enroll-upload-zone');
    const enrollPlaceholder = document.getElementById('enroll-upload-placeholder');
    const enrollFileInfo = document.getElementById('enroll-file-info');
    const enrollFileName = document.getElementById('enroll-file-name');
    const enrollBtn = document.getElementById('enroll-btn');
    const profileStatus = document.getElementById('profile-status');
    const profileHashDisplay = document.getElementById('profile-hash-display');

    // Monitoring Elements
    const monitorInput = document.getElementById('monitor-audio-input');
    const monitorZone = document.getElementById('monitor-upload-zone');
    const monitorPlaceholder = document.getElementById('monitor-upload-placeholder');
    const monitorFileInfo = document.getElementById('monitor-file-info');
    const monitorFileName = document.getElementById('monitor-file-name');
    const monitorBtn = document.getElementById('monitor-btn');
    
    // Mode Elements
    const modeUploadBtn = document.getElementById('mode-upload-btn');
    const modeSimBtn = document.getElementById('mode-sim-btn');
    const modeLiveBtn = document.getElementById('mode-live-btn');
    const monitorUploadSection = document.getElementById('monitor-upload-section');
    const monitorSimSection = document.getElementById('monitor-sim-section');
    const monitorLiveSection = document.getElementById('monitor-live-section');
    
    // Simulate Call Elements
    const simInput = document.getElementById('sim-audio-input');
    const simZone = document.getElementById('sim-upload-zone');
    const simPlaceholder = document.getElementById('sim-upload-placeholder');
    const simFileInfo = document.getElementById('sim-file-info');
    const simFileName = document.getElementById('sim-file-name');
    const startSimBtn = document.getElementById('start-sim-btn');
    const stopSimBtn = document.getElementById('stop-sim-btn');
    const simStatusBox = document.getElementById('sim-status-box');
    const simStatusText = document.getElementById('sim-status-text');
    let simFile = null;
    let simSocket = null;

    // Live Monitoring Elements
    const startLiveBtn = document.getElementById('start-live-btn');
    const stopLiveBtn = document.getElementById('stop-live-btn');
    const liveIndicator = document.getElementById('live-indicator');
    const liveStatusText = document.getElementById('live-status-text');
    
    let mediaRecorder = null;
    let liveSocket = null;
    let audioStream = null;
    
    // Results
    const resultCard = document.getElementById('result-card');
    const ringProgress = document.getElementById('ring-progress');
    const riskPercent = document.getElementById('risk-percent');
    const verdictTitle = document.getElementById('verdict-title');
    const verdictSubtitle = document.getElementById('verdict-subtitle');

    // --- Init ---
    if (token) {
        checkAuth();
    }

    // --- Auth Logic ---
    authToggleBtn.addEventListener('click', (e) => {
        e.preventDefault();
        isLoginMode = !isLoginMode;
        authTitle.textContent = isLoginMode ? 'Sign In' : 'Register';
        authSubmitBtn.textContent = isLoginMode ? 'Login' : 'Create Account';
        authToggleText.textContent = isLoginMode ? "Don't have an account?" : "Already have an account?";
        authToggleBtn.textContent = isLoginMode ? 'Register' : 'Login';
        authError.classList.add('hidden');
    });

    authForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = document.getElementById('auth-email').value;
        const password = document.getElementById('auth-password').value;
        
        authSubmitBtn.disabled = true;
        authError.classList.add('hidden');

        try {
            if (isLoginMode) {
                // Login
                const fd = new FormData();
                fd.append('username', email);
                fd.append('password', password);
                
                const res = await fetch('/api/auth/login', {
                    method: 'POST',
                    body: fd
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail);
                
                token = data.access_token;
                localStorage.setItem('token', token);
                checkAuth();
            } else {
                // Register
                const fd = new FormData();
                fd.append('email', email);
                fd.append('password', password);
                
                const res = await fetch('/api/auth/register', {
                    method: 'POST',
                    body: fd
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail);
                
                // auto login after register
                isLoginMode = true;
                authForm.dispatchEvent(new Event('submit'));
            }
        } catch (err) {
            authError.textContent = err.message;
            authError.classList.remove('hidden');
        } finally {
            authSubmitBtn.disabled = false;
        }
    });

    async function checkAuth() {
        try {
            const res = await fetch('/api/auth/me', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (!res.ok) throw new Error('Invalid token');
            
            const user = await res.json();
            userEmailDisplay.textContent = user.email;
            authView.classList.add('hidden');
            dashboardView.classList.remove('hidden');
            userStatus.classList.remove('hidden');
            
            checkProfile();
        } catch (e) {
            token = null;
            localStorage.removeItem('token');
            authView.classList.remove('hidden');
            dashboardView.classList.add('hidden');
            userStatus.classList.add('hidden');
        }
    }

    document.getElementById('logout-btn').addEventListener('click', () => {
        token = null;
        localStorage.removeItem('token');
        location.reload();
    });

    async function checkProfile() {
        try {
            const res = await fetch('/api/voice/profile', {
                headers: { 'Authorization': `Bearer ${token}` }
            });
            if (res.ok) {
                const profile = await res.json();
                profileStatus.classList.remove('hidden');
                profileHashDisplay.textContent = profile.profile_hash;
            }
        } catch(e) {}
    }

    // --- Tabs Logic ---
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active', 'hidden'));
            tabContents.forEach(c => c.classList.add('hidden'));
            
            btn.classList.add('active');
            document.getElementById(btn.dataset.target).classList.remove('hidden');
            document.getElementById(btn.dataset.target).classList.add('active');
        });
    });

    // --- Enrollment Logic ---
    enrollPlaceholder.addEventListener('click', () => enrollInput.click());
    enrollInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            enrollFile = e.target.files[0];
            enrollPlaceholder.classList.add('hidden');
            enrollFileInfo.classList.remove('hidden');
            enrollFileName.textContent = enrollFile.name;
            enrollBtn.disabled = false;
        }
    });

    enrollBtn.addEventListener('click', async () => {
        if(!enrollFile) return;
        enrollBtn.disabled = true;
        enrollBtn.textContent = "ENROLLING...";
        
        const fd = new FormData();
        fd.append('file', enrollFile);
        
        try {
            const res = await fetch('/api/voice/enrollment', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` },
                body: fd
            });
            const data = await res.json();
            if(!res.ok) throw new Error(data.detail || data.message);
            
            alert(data.message);
            checkProfile();
            
            // clear form
            enrollFile = null;
            enrollFileInfo.classList.add('hidden');
            enrollPlaceholder.classList.remove('hidden');
        } catch (e) {
            alert("Error: " + e.message);
        } finally {
            enrollBtn.disabled = false;
            enrollBtn.textContent = "ENROLL VOICE";
        }
    });

    // --- Monitoring Logic ---
    monitorPlaceholder.addEventListener('click', () => monitorInput.click());
    monitorInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            monitorFile = e.target.files[0];
            monitorPlaceholder.classList.add('hidden');
            monitorFileInfo.classList.remove('hidden');
            monitorFileName.textContent = monitorFile.name;
            monitorBtn.disabled = false;
            resultCard.classList.add('hidden');
        }
    });

    monitorBtn.addEventListener('click', async () => {
        if(!monitorFile) return;
        
        // Reset UI
        resultCard.classList.remove('hidden');
        verdictTitle.textContent = "Analyzing Call...";
        verdictSubtitle.textContent = "Checking for synthetic voice characteristics";
        riskPercent.textContent = "0%";
        ringProgress.style.strokeDashoffset = 339.292;
        ringProgress.classList.remove('stroke-red', 'stroke-cyan', 'stroke-amber');
        verdictTitle.classList.remove('text-red', 'text-cyan', 'text-amber');
        monitorBtn.disabled = true;
        
        const fd = new FormData();
        fd.append('audio_file', monitorFile);
        
        try {
            const res = await fetch('/api/monitoring/sessions', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` },
                body: fd
            });
            const data = await res.json();
            if(!res.ok) throw new Error(data.detail);
            
            displayResult(data);
        } catch (e) {
            verdictTitle.textContent = "Detection Failed";
            verdictSubtitle.textContent = e.message;
            verdictTitle.classList.add('text-red');
        } finally {
            monitorBtn.disabled = false;
        }
    });
    
    // --- Monitoring Mode Switch ---
    function setActiveMode(activeBtn, activeSection) {
        [modeUploadBtn, modeSimBtn, modeLiveBtn].forEach(b => b.classList.remove('active'));
        [monitorUploadSection, monitorSimSection, monitorLiveSection].forEach(s => s.classList.add('hidden'));
        activeBtn.classList.add('active');
        activeSection.classList.remove('hidden');
        stopLiveMonitoring();
        stopSimulation();
    }
    
    modeUploadBtn.addEventListener('click', () => setActiveMode(modeUploadBtn, monitorUploadSection));
    modeSimBtn.addEventListener('click', () => setActiveMode(modeSimBtn, monitorSimSection));
    modeLiveBtn.addEventListener('click', () => setActiveMode(modeLiveBtn, monitorLiveSection));

    // --- Simulate Call: File Selection ---
    simZone.addEventListener('click', () => simInput.click());
    simInput.addEventListener('change', (e) => {
        if(e.target.files.length) {
            simFile = e.target.files[0];
            simPlaceholder.classList.add('hidden');
            simFileInfo.classList.remove('hidden');
            simFileName.textContent = simFile.name;
            startSimBtn.disabled = false;
        }
    });

    // --- Simulate Call: Stream File Through WebSocket ---
    startSimBtn.addEventListener('click', async () => {
        if(!simFile) return;
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        simSocket = new WebSocket(`${protocol}//${window.location.host}/api/monitoring/simulate?token=${token}`);
        
        simSocket.onopen = async () => {
            startSimBtn.classList.add('hidden');
            stopSimBtn.classList.remove('hidden');
            simStatusBox.style.display = 'flex';
            simStatusText.textContent = 'Uploading audio file...';
            resultCard.classList.remove('hidden');
            verdictTitle.textContent = "Analyzing Call...";
            verdictSubtitle.textContent = "Sending file to server for chunked analysis...";
            
            // Send the raw file bytes — server handles chunking
            const arrayBuffer = await simFile.arrayBuffer();
            simSocket.send(arrayBuffer);
            simStatusText.textContent = 'Server is analyzing chunks...';
        };
        
        simSocket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if(data.error) {
                stopSimulation();
                alert(data.error);
                return;
            }
            if(data.status === 'processing') {
                simStatusText.textContent = data.message;
                return;
            }
            if(data.status === 'complete') {
                simStatusText.textContent = `✅ Analysis complete! (${data.total_chunks} chunks)`;
                displayResult(data);
                return;
            }
            displayResult(data);
            simStatusText.textContent = `Chunk ${data.chunk}/${data.total_chunks} | This chunk: ${data.chunk_ai}% AI`;
        };
        
        simSocket.onclose = () => {
            // Don't reset UI here, let stopSimulation handle it
        };
    });
    
    stopSimBtn.addEventListener('click', stopSimulation);
    
    function stopSimulation() {
        if(simSocket) {
            try { simSocket.close(); } catch(e) {}
            simSocket = null;
        }
        stopSimBtn.classList.add('hidden');
        startSimBtn.classList.remove('hidden');
        startSimBtn.disabled = !simFile;
    }

    // --- Live Monitoring Logic ---
    startLiveBtn.addEventListener('click', async () => {
        try {
            audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Connect WebSocket
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            liveSocket = new WebSocket(`${protocol}//${window.location.host}/api/monitoring/live?token=${token}`);
            
            liveSocket.onopen = () => {
                liveIndicator.classList.add('recording');
                liveStatusText.textContent = 'Listening and Analyzing...';
                startLiveBtn.classList.add('hidden');
                stopLiveBtn.classList.remove('hidden');
                resultCard.classList.remove('hidden');
                verdictTitle.textContent = "Analyzing Call...";
                verdictSubtitle.textContent = "Awaiting first chunk...";
                
                mediaRecorder = new MediaRecorder(audioStream);
                mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0 && liveSocket.readyState === WebSocket.OPEN) {
                        liveSocket.send(e.data);
                    }
                };
                mediaRecorder.start(4000); // 4-second chunks
            };
            
            liveSocket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if(data.error) {
                    stopLiveMonitoring();
                    alert(data.error);
                    return;
                }
                displayResult(data);
            };
            
            liveSocket.onclose = () => {
                stopLiveMonitoring();
            };
            
        } catch (e) {
            alert('Microphone access denied or error occurred: ' + e.message);
        }
    });
    
    stopLiveBtn.addEventListener('click', stopLiveMonitoring);
    
    function stopLiveMonitoring() {
        if(mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        if(audioStream) {
            audioStream.getTracks().forEach(t => t.stop());
        }
        if(liveSocket) {
            liveSocket.close();
        }
        
        liveIndicator.classList.remove('recording');
        liveStatusText.textContent = 'Microphone Idle';
        stopLiveBtn.classList.add('hidden');
        startLiveBtn.classList.remove('hidden');
    }

    function displayResult(data) {
        const circumference = 339.292;
        const offset = circumference - (data.risk_score / 100) * circumference;
        ringProgress.style.strokeDashoffset = offset;
        ringProgress.style.transition = 'stroke-dashoffset 0.5s ease';
        
        riskPercent.textContent = Math.round(data.risk_score) + '%';
        
        let colorClass = 'cyan';
        let title = 'SECURE CALL';
        
        if (data.prediction === 'FAKE' || data.prediction === 'AI_GENERATED') {
            colorClass = 'red';
            title = 'HIGH RISK CALL';
        } else if (data.prediction === 'UNKNOWN' || data.prediction === 'UNCERTAIN') {
            colorClass = 'amber';
            title = 'UNCERTAIN';
        } else if (data.prediction === 'ERROR') {
            colorClass = 'amber';
            title = 'ANALYSIS ERROR';
            data.risk_score = 0;
            riskPercent.textContent = '---';
        }
        
        ringProgress.classList.remove('stroke-red', 'stroke-cyan', 'stroke-amber');
        verdictTitle.classList.remove('text-red', 'text-cyan', 'text-amber');
        
        ringProgress.classList.add(`stroke-${colorClass}`);
        verdictTitle.classList.add(`text-${colorClass}`);
        verdictTitle.textContent = title;
        
        if (data.prediction === 'ERROR') {
            verdictSubtitle.textContent = data.details && data.details.length > 0 ? data.details[0] : "Need more audio data... Keep speaking.";
        } else if (data.chunk) {
            verdictSubtitle.textContent = `Chunk #${data.chunk} | This chunk: ${data.chunk_ai}% AI | Avg: ${data.confidence}%`;
        } else {
            verdictSubtitle.textContent = `Confidence: ${data.confidence}%`;
        }
    }
});
