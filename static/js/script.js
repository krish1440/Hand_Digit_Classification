document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('digitCanvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clearBtn');
    const predictBtn = document.getElementById('predictBtn');
    const predictionResult = document.getElementById('predictionResult');
    const confidenceResult = document.getElementById('confidenceResult');
    
    let isDrawing = false;
    let chart = null;

    // --- Chart Setup ---
    function initChart() {
        const ctxChart = document.getElementById('probChart').getContext('2d');
        chart = new Chart(ctxChart, {
            type: 'bar',
            data: {
                labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                datasets: [{
                    label: 'Probability',
                    data: Array(10).fill(0),
                    backgroundColor: 'rgba(99, 102, 241, 0.6)',
                    borderColor: 'rgba(99, 102, 241, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    // --- Canvas Logic ---
    // Setup canvas drawing style (Black on White for user logic? 
    // Wait, script says "Canvas is black on white", but preprocessing does "255 - img" (Invert).
    // So user should draw Black on White.
    
    // Fill white background first
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 20; // Thicker line for better recognition after resizing
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    function getPos(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }

    function startDraw(e) {
        e.preventDefault();
        isDrawing = true;
        const pos = getPos(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const pos = getPos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    }

    function stopDraw() {
        isDrawing = false;
    }

    // Mouse Events
    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDraw);
    canvas.addEventListener('mouseout', stopDraw);

    // Touch Events
    canvas.addEventListener('touchstart', startDraw);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('touchend', stopDraw);

    // --- Buttons ---
    clearBtn.addEventListener('click', () => {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        predictionResult.textContent = '-';
        confidenceResult.textContent = 'Confidence: -%';
        updateChart(Array(10).fill(0));
    });

    predictBtn.addEventListener('click', async () => {
        // Get image data URL
        const dataURL = canvas.toDataURL('image/png');
        
        try {
            predictBtn.textContent = 'Analyzing...';
            predictBtn.disabled = true;

            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataURL })
            });

            const data = await response.json();

            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            // Update UI
            predictionResult.textContent = data.digit;
            confidenceResult.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
            updateChart(data.probabilities);

        } catch (error) {
            console.error('Error:', error);
            alert('Something went wrong!');
        } finally {
            predictBtn.textContent = 'Predict';
            predictBtn.disabled = false;
        }
    });

    function updateChart(data) {
        if (chart) {
            chart.data.datasets[0].data = data;
            chart.update();
        }
    }

    initChart();
});
