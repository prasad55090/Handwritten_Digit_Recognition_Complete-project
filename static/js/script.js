document.addEventListener("DOMContentLoaded", () => {
    // -----------------------------------------------------------------
    // 1. Theme Configuration & Mode Toggles
    // -----------------------------------------------------------------
    const themeToggleBtn = document.getElementById("themeToggle");
    const currentTheme = localStorage.getItem("theme") || "light"; // Academic light theme default
    document.documentElement.setAttribute("data-theme", currentTheme);
    updateThemeIcon(currentTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener("click", () => {
            const activeTheme = document.documentElement.getAttribute("data-theme");
            const newTheme = activeTheme === "dark" ? "light" : "dark";
            document.documentElement.setAttribute("data-theme", newTheme);
            localStorage.setItem("theme", newTheme);
            updateThemeIcon(newTheme);
            showToast("Theme updated", "info");
        });
    }

    function updateThemeIcon(theme) {
        if (!themeToggleBtn) return;
        const icon = themeToggleBtn.querySelector("i");
        icon.className = theme === "dark" ? "fas fa-sun" : "fas fa-moon";
    }

    // -----------------------------------------------------------------
    // 2. Toast Alerts Helper
    // -----------------------------------------------------------------
    const toastContainer = document.getElementById("toastContainer");

    window.showToast = function(message, type = "success") {
        if (!toastContainer) return;
        const toast = document.createElement("div");
        toast.className = `toast-custom ${type}`;
        
        let icon = "info-circle";
        if (type === "success") icon = "check-circle text-success";
        if (type === "error") icon = "exclamation-triangle text-danger";
        
        toast.innerHTML = `
            <i class="fas fa-${icon}"></i>
            <span>${message}</span>
        `;
        
        toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.animation = "slideIn 0.25s ease-out reverse forwards";
            setTimeout(() => toast.remove(), 250);
        }, 3000);
    }

    // -----------------------------------------------------------------
    // 3. Loader Overlay Helper
    // -----------------------------------------------------------------
    const loader = document.getElementById("loadingOverlay");
    
    function showLoader() {
        if (loader) loader.style.display = "flex";
    }

    function hideLoader() {
        if (loader) loader.style.display = "none";
    }

    // -----------------------------------------------------------------
    // 4. Chart.js Helper Initialization
    // -----------------------------------------------------------------
    const ctxChart = document.getElementById("probabilityChart");
    let probChart = null;

    function initChart() {
        if (!ctxChart) return;
        const isDark = document.documentElement.getAttribute("data-theme") === "dark";
        const gridColor = isDark ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.05)";
        const tickColor = isDark ? "#94a3b8" : "#475569";

        probChart = new Chart(ctxChart, {
            type: 'bar',
            data: {
                labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
                datasets: [{
                    label: 'Confidence (%)',
                    data: Array(36).fill(0),
                    backgroundColor: 'rgba(37, 99, 235, 0.65)',
                    borderColor: 'rgba(37, 99, 235, 1)',
                    borderWidth: 1.5,
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: gridColor },
                        ticks: { color: tickColor, font: { family: 'Outfit' } }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: tickColor, font: { family: 'Outfit', weight: 'bold', size: 11 } }
                    }
                }
            }
        });
    }

    function updateChartData(probabilities) {
        if (!probChart) return;
        const paired = probabilities.map((p, idx) => ({ 
            prob: p * 100, 
            label: probChart.data.labels[idx] || idx.toString() 
        }));
        
        paired.sort((a, b) => b.prob - a.prob);
        const topPaired = paired.slice(0, 6); // Top 6 candidates
        
        probChart.data.labels = topPaired.map(p => p.label);
        probChart.data.datasets[0].data = topPaired.map(p => p.prob);
        probChart.update();
    }

    initChart();

    // -----------------------------------------------------------------
    // 5. Canvas Drawing Board
    // -----------------------------------------------------------------
    const canvas = document.getElementById("paintCanvas");
    if (canvas) {
        const ctx = canvas.getContext("2d");
        const clearBtn = document.getElementById("clearCanvasBtn");
        const predictBtn = document.getElementById("predictCanvasBtn");
        const brushSizeSlider = document.getElementById("brushSize");
        
        let drawing = false;
        let lastX = 0;
        let lastY = 0;
        let canvasEdited = false;

        function resetCanvas() {
            ctx.fillStyle = "#000000";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            canvasEdited = false;
            resetPredictionDisplay();
            showToast("Canvas cleared", "info");
        }

        resetCanvas();

        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.strokeStyle = "#FFFFFF"; // Drawing in white ink on black grid
        
        function getBrushWidth() {
            return brushSizeSlider ? parseInt(brushSizeSlider.value) : 18;
        }

        canvas.addEventListener("mousedown", (e) => {
            drawing = true;
            canvasEdited = true;
            [lastX, lastY] = [e.offsetX, e.offsetY];
        });

        canvas.addEventListener("mousemove", (e) => {
            if (!drawing) return;
            ctx.lineWidth = getBrushWidth();
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.stroke();
            [lastX, lastY] = [e.offsetX, e.offsetY];
        });

        window.addEventListener("mouseup", () => {
            if (drawing) {
                drawing = false;
                autoPredictCanvas();
            }
        });

        canvas.addEventListener("touchstart", (e) => {
            e.preventDefault();
            drawing = true;
            canvasEdited = true;
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            lastX = touch.clientX - rect.left;
            lastY = touch.clientY - rect.top;
        });

        canvas.addEventListener("touchmove", (e) => {
            e.preventDefault();
            if (!drawing) return;
            const touch = e.touches[0];
            const rect = canvas.getBoundingClientRect();
            const x = touch.clientX - rect.left;
            const y = touch.clientY - rect.top;

            ctx.lineWidth = getBrushWidth();
            ctx.beginPath();
            ctx.moveTo(lastX, lastY);
            ctx.lineTo(x, y);
            ctx.stroke();
            [lastX, lastY] = [x, y];
        });

        window.addEventListener("touchend", () => {
            if (drawing) {
                drawing = false;
                autoPredictCanvas();
            }
        });

        if (clearBtn) clearBtn.addEventListener("click", resetCanvas);
        if (predictBtn) predictBtn.addEventListener("click", submitCanvasForPrediction);

        function autoPredictCanvas() {
            // Checks if auto-prediction is toggled inside localStorage settings
            const autoEnabled = localStorage.getItem("setting_auto_predict") !== "false";
            if (autoEnabled && canvasEdited) {
                submitCanvasForPrediction();
            }
        }

        function submitCanvasForPrediction() {
            if (!canvasEdited) return;
            const base64Data = canvas.toDataURL("image/png");
            sendPredictionRequest(base64Data, "Canvas Drawing");
        }
    }

    // -----------------------------------------------------------------
    // 6. Image File Upload Section
    // -----------------------------------------------------------------
    const dragZone = document.getElementById("uploadDragZone");
    const fileInput = document.getElementById("imageFileInput");
    const uploadPreview = document.getElementById("uploadPreview");
    const uploadPlaceholder = document.getElementById("uploadPlaceholder");

    if (dragZone && fileInput) {
        dragZone.addEventListener("click", () => fileInput.click());

        dragZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dragZone.classList.add("dragover");
        });

        ["dragleave", "dragend"].forEach(type => {
            dragZone.addEventListener(type, () => dragZone.classList.remove("dragover"));
        });

        dragZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dragZone.classList.remove("dragover");
            const files = e.dataTransfer.files;
            if (files.length) {
                fileInput.files = files;
                handleUploadFile(files[0]);
            }
        });

        fileInput.addEventListener("change", (e) => {
            if (e.target.files.length) {
                handleUploadFile(e.target.files[0]);
            }
        });

        function handleUploadFile(file) {
            if (!file.type.match('image.*')) {
                showToast("Please upload an image!", "error");
                return;
            }

            const reader = new FileReader();
            reader.onload = (e) => {
                uploadPreview.src = e.target.result;
                uploadPreview.classList.remove("d-none");
                uploadPlaceholder.classList.add("d-none");
                sendPredictionRequest(e.target.result, "Image Upload");
            };
            reader.readAsDataURL(file);
        }

        const clearUploadBtn = document.getElementById("clearUploadBtn");
        if (clearUploadBtn) {
            clearUploadBtn.addEventListener("click", () => {
                fileInput.value = "";
                uploadPreview.src = "";
                uploadPreview.classList.add("d-none");
                uploadPlaceholder.classList.remove("d-none");
                resetPredictionDisplay();
                showToast("Upload cleared", "info");
            });
        }
    }

    // -----------------------------------------------------------------
    // 7. Prediction Results & Dynamic Routing
    // -----------------------------------------------------------------
    let lastPredictionBase64 = null;

    function updatePipelineStatus(state, message, executionTime = null) {
        const statusBadge = document.getElementById("processingStatus");
        const statusText = document.getElementById("statusText");
        const statusMessage = document.getElementById("statusMessage");
        const spinner = document.getElementById("statusIndicatorSpinner");
        const icon = document.getElementById("statusIndicatorIcon");
        const timeEl = document.getElementById("predTime");

        if (!statusBadge) return; // Only runs on pages with the status panel (upload.html)

        // Reset badge classes
        statusBadge.className = "badge px-3 py-2 fs-6";
        
        if (state === "loading") {
            statusBadge.classList.add("bg-warning", "text-dark");
            statusBadge.innerText = "Status: Loading...";
            
            if (statusText) statusText.innerText = "Processing Pipeline Active";
            if (statusMessage) statusMessage.innerText = message || "Applying thresholding and running CNN/TrOCR models...";
            
            if (spinner) spinner.classList.remove("d-none");
            if (icon) {
                icon.className = "d-none";
            }
            if (timeEl) timeEl.innerText = "Running...";
        } else if (state === "completed") {
            statusBadge.classList.add("bg-success");
            statusBadge.innerText = "Status: Completed";
            
            if (statusText) statusText.innerText = "Processing Completed Successfully!";
            if (statusMessage) statusMessage.innerText = message || "OCR extraction completed.";
            
            if (spinner) spinner.classList.add("d-none");
            if (icon) {
                icon.className = "fas fa-check-circle text-success";
                icon.classList.remove("d-none");
            }
            if (timeEl) timeEl.innerText = executionTime || "N/A";
        } else if (state === "error") {
            statusBadge.classList.add("bg-danger");
            statusBadge.innerText = "Status: Error";
            
            if (statusText) statusText.innerText = "Pipeline Failed";
            if (statusMessage) statusMessage.innerText = message || "An error occurred during prediction.";
            
            if (spinner) spinner.classList.add("d-none");
            if (icon) {
                icon.className = "fas fa-exclamation-triangle text-danger";
                icon.classList.remove("d-none");
            }
            if (timeEl) timeEl.innerText = "Failed";
        } else { // idle
            statusBadge.classList.add("bg-secondary");
            statusBadge.innerText = "Status: Idle";
            
            if (statusText) statusText.innerText = "Ready to Process";
            if (statusMessage) statusMessage.innerText = "Upload or drag/drop a handwritten document image on the left to start.";
            
            if (spinner) spinner.classList.add("d-none");
            if (icon) {
                icon.className = "fas fa-info-circle text-muted";
                icon.classList.remove("d-none");
            }
            if (timeEl) timeEl.innerText = "-";
        }
    }

    function sendPredictionRequest(base64Image, sourceName) {
        showLoader();
        updatePipelineStatus("loading", "Extracting base64 payload, running adaptive deskewing, super resolution, and layout character segmentation...");
        lastPredictionBase64 = base64Image;

        // Fetch settings from local storage to include
        const threshold = localStorage.getItem("setting_threshold") || "127";
        const manualModel = localStorage.getItem("setting_model_mode") || "auto";

        fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                image: base64Image, 
                source: sourceName,
                threshold: parseInt(threshold),
                model_mode: manualModel
            })
        })
        .then(async res => {
            const isJson = res.headers.get('content-type')?.includes('application/json');
            const data = isJson ? await res.json() : null;
            if (!res.ok) {
                const errMsg = (data && data.error) ? data.error : `HTTP error ${res.status}`;
                throw new Error(errMsg);
            }
            return data;
        })
        .then(data => {
            hideLoader();

            // Update Pipeline Status panel
            updatePipelineStatus("completed", `OCR text successfully extracted and logged to SQLite history!`, data.prediction_time);

            // Save small metadata only to sessionStorage to prevent storage quota exceed errors
            sessionStorage.setItem("last_pred_text", data.text);
            sessionStorage.setItem("last_pred_confidence", data.confidence);
            sessionStorage.setItem("last_pred_type", data.input_type);
            sessionStorage.setItem("last_pred_timestamp", new Date().toLocaleString());

            // If we are currently on Draw or Upload, populate local elements
            updateUIDisplay(data);

            // Redirect automatically if "Automatic Redirect to Prediction Page" setting is enabled
            // Only redirect for the drawing canvas, disable for same-page upload results
            const autoRedirect = localStorage.getItem("setting_auto_redirect") === "true";
            if (autoRedirect && window.location.pathname.includes("draw")) {
                window.location.href = "/prediction";
            }
        })
        .catch(err => {
            hideLoader();
            console.error("HTR Predict Dispatch Error:", err);
            updatePipelineStatus("error", `Prediction failed: ${err.message}`);
            const extractedTextArea = document.getElementById("extractedTextArea");
            if (extractedTextArea) {
                extractedTextArea.value = `Error: ${err.message}`;
            }
            showToast(err.message, "error");
        });
    }

    function updateUIDisplay(data) {
        const textEl = document.getElementById("predText");
        const confEl = document.getElementById("predConfidence");
        const typeEl = document.getElementById("predInputType");
        const segContainer = document.getElementById("segmentationPreviewContainer");
        const extractedTextArea = document.getElementById("extractedTextArea");

        if (textEl) {
            textEl.innerText = data.text;
            textEl.classList.remove("empty");
        }
        if (confEl) {
            confEl.innerText = `${(data.confidence * 100).toFixed(2)}% Confidence`;
        }
        if (typeEl) {
            typeEl.innerText = `Type: ${data.input_type.toUpperCase()}`;
        }
        if (extractedTextArea) {
            extractedTextArea.value = data.text;
        }

        if (segContainer) {
            if (data.segments && data.segments.length > 0) {
                segContainer.innerHTML = data.segments.map(seg => `
                    <div class="segmented-character-card">
                        <img src="${seg.image}" alt="Segment">
                        <span class="char-val">${seg.char.split(" Box:")[0]}</span>
                    </div>
                `).join('');
            } else {
                segContainer.innerHTML = `<span class="text-muted small">No segment boxes generated</span>`;
            }
        }

        if (data.probabilities) {
            updateChartData(data.probabilities);
        }
    }

    function resetPredictionDisplay() {
        const textEl = document.getElementById("predText");
        const confEl = document.getElementById("predConfidence");
        const typeEl = document.getElementById("predInputType");
        const segContainer = document.getElementById("segmentationPreviewContainer");
        const extractedTextArea = document.getElementById("extractedTextArea");

        if (textEl && confEl) {
            textEl.innerText = "-";
            textEl.classList.add("empty");
            confEl.innerText = "Confidence: -";
            if (typeEl) typeEl.innerText = "Type: -";
        }
        
        if (extractedTextArea) {
            extractedTextArea.value = "";
        }

        if (segContainer) {
            segContainer.innerHTML = `<span class="text-muted small">No active segments</span>`;
        }

        if (probChart) {
            probChart.data.datasets[0].data = Array(36).fill(0);
            probChart.update();
        }

        updatePipelineStatus("idle");
    }

    // Direct actions copy & download for upload page
    const copyBtnDirect = document.getElementById("copyTextBtnDirect");
    if (copyBtnDirect) {
        copyBtnDirect.addEventListener("click", () => {
            const txt = document.getElementById("extractedTextArea")?.value;
            if (!txt || txt === "") {
                showToast("No text to copy", "error");
                return;
            }
            navigator.clipboard.writeText(txt)
                .then(() => showToast("Copied to clipboard", "success"))
                .catch(() => showToast("Copy failed", "error"));
        });
    }

    const downloadTxtBtnDirect = document.getElementById("downloadTxtBtnDirect");
    if (downloadTxtBtnDirect) {
        downloadTxtBtnDirect.addEventListener("click", (e) => {
            e.preventDefault();
            const txt = document.getElementById("extractedTextArea")?.value;
            if (!txt || txt === "") {
                showToast("No text to export", "error");
                return;
            }
            const blob = new Blob([txt], { type: "text/plain;charset=utf-8" });
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = `htr-extracted-text.txt`;
            link.click();
            showToast("TXT file downloaded", "success");
        });
    }

    const downloadDocBtnDirect = document.getElementById("downloadDocBtnDirect");
    if (downloadDocBtnDirect) {
        downloadDocBtnDirect.addEventListener("click", (e) => {
            e.preventDefault();
            const txt = document.getElementById("extractedTextArea")?.value;
            const conf = document.getElementById("predConfidence")?.innerText;
            const type = document.getElementById("predInputType")?.innerText;
            if (!txt || txt === "") {
                showToast("No text to export", "error");
                return;
            }

            const htmlContent = `
                <html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
                <head>
                    <title>Handwritten OCR Prediction</title>
                    <style>
                        body { font-family: Arial, sans-serif; padding: 20px; line-height: 1.6; }
                        h2 { color: #1e3a8a; }
                        .text-block { background-color: #f1f5f9; padding: 15px; border-radius: 8px; border: 1px solid #cbd5e1; font-family: Courier, monospace; }
                        .meta { color: #64748b; font-size: 12px; margin-top: 20px; }
                    </style>
                </head>
                <body>
                    <h2>Handwritten Text Recognition Report</h2>
                    <p><strong>Execution Date:</strong> ${new Date().toLocaleString()}</p>
                    <p><strong>Accuracy Index:</strong> ${conf}</p>
                    <p><strong>Classified Type:</strong> ${type}</p>
                    <hr/>
                    <h3>Extracted Text:</h3>
                    <div class="text-block">${txt.replace(/\n/g, "<br>")}</div>
                    <div class="meta">Generated directly from dynamic HTR portal. MCA Capstone Project.</div>
                </body>
                </html>
            `;
            const blob = new Blob(['\ufeff' + htmlContent], { type: 'application/msword' });
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = `htr-extracted-text.doc`;
            link.click();
            showToast("Word document downloaded", "success");
        });
    }

    const downloadPdfBtnDirect = document.getElementById("downloadPdfBtnDirect");
    if (downloadPdfBtnDirect) {
        downloadPdfBtnDirect.addEventListener("click", (e) => {
            e.preventDefault();
            const txt = document.getElementById("extractedTextArea")?.value;
            const conf = document.getElementById("predConfidence")?.innerText;
            const type = document.getElementById("predInputType")?.innerText;
            
            if (!txt || txt === "") {
                showToast("No text to export", "error");
                return;
            }

            const { jsPDF } = window.jspdf;
            const doc = new jsPDF();
            
            doc.setFillColor(30, 58, 138); // Academic navy blue
            doc.rect(0, 0, 210, 40, "F");
            
            doc.setTextColor(255, 255, 255);
            doc.setFont("Helvetica", "bold");
            doc.setFontSize(20);
            doc.text("Handwritten OCR System", 20, 26);
            
            doc.setTextColor(100, 116, 139);
            doc.setFont("Helvetica", "normal");
            doc.setFontSize(10);
            doc.text(`Timestamp: ${new Date().toLocaleString()}`, 20, 52);
            doc.text(`Execution Context: ${type}`, 20, 58);
            doc.text(`Model Accuracy Rating: ${conf}`, 20, 64);
            
            doc.setDrawColor(226, 232, 240);
            doc.line(20, 70, 190, 70);
            
            doc.setTextColor(30, 58, 138);
            doc.setFontSize(14);
            doc.text("Extracted Handwritten Text:", 20, 84);
            
            doc.setTextColor(30, 41, 59);
            doc.setFont("Courier", "normal");
            doc.setFontSize(11);
            
            const lines = doc.splitTextToSize(txt, 170);
            doc.text(lines, 20, 95);
            
            doc.save(`htr-ocr-pdf-report.pdf`);
            showToast("PDF report downloaded", "success");
        });
    }

    // -----------------------------------------------------------------
    // 8. Prediction Detail Page Hydrator
    // -----------------------------------------------------------------
    if (window.location.pathname.includes("prediction")) {
        showLoader();
        fetch("/api/latest-prediction")
        .then(async res => {
            const isJson = res.headers.get('content-type')?.includes('application/json');
            const data = isJson ? await res.json() : null;
            if (!res.ok) {
                const errMsg = (data && data.error) ? data.error : `HTTP error ${res.status}`;
                throw new Error(errMsg);
            }
            return data;
        })
        .then(data => {
            hideLoader();
            const detailText = document.getElementById("detailPredText");
            const detailConf = document.getElementById("detailPredConfidence");
            const detailType = document.getElementById("detailPredType");
            const detailImg = document.getElementById("detailInputImage");
            
            if (detailText) detailText.innerText = data.text;
            if (detailConf) detailConf.innerText = `${(parseFloat(data.confidence) * 100).toFixed(2)}% Confidence`;
            if (detailType) detailType.innerText = `Classified Input: ${data.input_type.toUpperCase()}`;
            
            if (detailImg && data.image) {
                detailImg.src = data.image;
                detailImg.classList.remove("d-none");
                const placeholder = document.getElementById("detailImagePlaceholder");
                if (placeholder) placeholder.classList.add("d-none");
            }

            // Populate segments
            const segContainer = document.getElementById("detailSegmentationContainer");
            if (segContainer && data.segments && data.segments.length > 0) {
                segContainer.innerHTML = data.segments.map(seg => `
                    <div class="segmented-character-card">
                        <img src="${seg.image}" alt="Segment">
                        <span class="char-val">${seg.char.split(" Box:")[0]}</span>
                    </div>
                `).join('');
            } else if (segContainer) {
                segContainer.innerHTML = `<span class="text-muted small">No active segments</span>`;
            }

            // Update Chart.js probability bars
            if (data.probabilities && data.probabilities.length > 0) {
                updateChartData(data.probabilities);
            }
        })
        .catch(err => {
            hideLoader();
            console.error("Failed to load prediction details:", err);
            showToast("Failed to load prediction details: " + err.message, "error");
        });
    }

    // -----------------------------------------------------------------
    // 9. Prediction Copy and Downloads (TXT, PDF, DOCX)
    // -----------------------------------------------------------------
    const copyBtn = document.getElementById("copyExtractedTextBtn");
    const downloadTxtBtn = document.getElementById("downloadTxtBtn");
    const downloadPdfBtn = document.getElementById("downloadPdfBtn");
    const downloadDocxBtn = document.getElementById("downloadDocxBtn");

    if (copyBtn) {
        copyBtn.addEventListener("click", () => {
            const txt = document.getElementById("detailPredText")?.innerText || document.getElementById("predText")?.innerText;
            if (!txt || txt === "-") {
                showToast("No text to copy", "error");
                return;
            }
            navigator.clipboard.writeText(txt)
                .then(() => showToast("Copied to clipboard", "success"))
                .catch(() => showToast("Copy failed", "error"));
        });
    }

    // Download TXT
    if (downloadTxtBtn) {
        downloadTxtBtn.addEventListener("click", () => {
            const txt = document.getElementById("detailPredText")?.innerText || document.getElementById("predText")?.innerText;
            if (!txt || txt === "-") return;
            
            const blob = new Blob([txt], { type: "text/plain;charset=utf-8" });
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = `htr-extracted-text.txt`;
            link.click();
            showToast("TXT file downloaded", "success");
        });
    }

    // Download DOC (renders HTML Word document matching DOCX capability client-side)
    if (downloadDocxBtn) {
        downloadDocxBtn.addEventListener("click", () => {
            const txt = document.getElementById("detailPredText")?.innerText || document.getElementById("predText")?.innerText;
            const conf = document.getElementById("detailPredConfidence")?.innerText || document.getElementById("predConfidence")?.innerText;
            if (!txt || txt === "-") return;

            const htmlContent = `
                <html xmlns:o='urn:schemas-microsoft-com:office:office' xmlns:w='urn:schemas-microsoft-com:office:word' xmlns='http://www.w3.org/TR/REC-html40'>
                <head>
                    <title>Handwritten OCR Prediction</title>
                    <style>
                        body { font-family: Arial, sans-serif; padding: 20px; line-height: 1.6; }
                        h2 { color: #1e3a8a; }
                        .text-block { background-color: #f1f5f9; padding: 15px; border-radius: 8px; border: 1px solid #cbd5e1; font-family: Courier, monospace; }
                        .meta { color: #64748b; font-size: 12px; margin-top: 20px; }
                    </style>
                </head>
                <body>
                    <h2>Handwritten Text Recognition Report</h2>
                    <p><strong>Execution Date:</strong> ${new Date().toLocaleString()}</p>
                    <p><strong>Accuracy Index:</strong> ${conf}</p>
                    <hr/>
                    <h3>Extracted Text:</h3>
                    <div class="text-block">${txt.replace(/\n/g, "<br>")}</div>
                    <div class="meta">Generated by Academic HTR Console. Final Year MCA Capstone Project.</div>
                </body>
                </html>
            `;
            const blob = new Blob(['\ufeff' + htmlContent], { type: 'application/msword' });
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = `htr-extracted-text.doc`;
            link.click();
            showToast("Word document downloaded", "success");
        });
    }

    // Download PDF (jsPDF compilation)
    if (downloadPdfBtn) {
        downloadPdfBtn.addEventListener("click", () => {
            const txt = document.getElementById("detailPredText")?.innerText || document.getElementById("predText")?.innerText;
            const conf = document.getElementById("detailPredConfidence")?.innerText || document.getElementById("predConfidence")?.innerText;
            const type = document.getElementById("detailPredType")?.innerText || document.getElementById("predInputType")?.innerText;
            
            if (!txt || txt === "-") return;

            const { jsPDF } = window.jspdf;
            const doc = new jsPDF();
            
            // Header
            doc.setFillColor(30, 58, 138); // Academic navy blue
            doc.rect(0, 0, 210, 40, "F");
            
            doc.setTextColor(255, 255, 255);
            doc.setFont("Helvetica", "bold");
            doc.setFontSize(20);
            doc.text("Handwritten OCR System", 20, 26);
            
            doc.setTextColor(100, 116, 139);
            doc.setFont("Helvetica", "normal");
            doc.setFontSize(10);
            doc.text(`Timestamp: ${new Date().toLocaleString()}`, 20, 52);
            doc.text(`Execution Context: ${type}`, 20, 58);
            doc.text(`Model Accuracy Rating: ${conf}`, 20, 64);
            
            doc.setDrawColor(226, 232, 240);
            doc.line(20, 70, 190, 70);
            
            doc.setTextColor(30, 58, 138);
            doc.setFontSize(14);
            doc.text("Extracted Handwritten Text:", 20, 84);
            
            doc.setTextColor(30, 41, 59);
            doc.setFont("Courier", "normal");
            doc.setFontSize(11);
            
            // Handle multiple lines text block mapping
            const lines = doc.splitTextToSize(txt, 170);
            doc.text(lines, 20, 95);
            
            doc.save(`htr-ocr-pdf-report.pdf`);
            showToast("PDF report downloaded", "success");
        });
    }

    // -----------------------------------------------------------------
    // 10. Settings Configuration Controls
    // -----------------------------------------------------------------
    if (window.location.pathname.includes("settings")) {
        const selectModel = document.getElementById("settingModelSelect");
        const sliderThresh = document.getElementById("settingThreshRange");
        const threshVal = document.getElementById("settingThreshVal");
        const toggleRedirect = document.getElementById("settingAutoRedirect");
        const toggleAutoPredict = document.getElementById("settingAutoPredict");
        
        // Load settings from local storage
        if (selectModel) selectModel.value = localStorage.getItem("setting_model_mode") || "auto";
        if (sliderThresh) {
            const storedThresh = localStorage.getItem("setting_threshold") || "127";
            sliderThresh.value = storedThresh;
            if (threshVal) threshVal.innerText = storedThresh;
        }
        if (toggleRedirect) toggleRedirect.checked = localStorage.getItem("setting_auto_redirect") === "true";
        if (toggleAutoPredict) toggleAutoPredict.checked = localStorage.getItem("setting_auto_predict") !== "false";

        // Save on change
        if (selectModel) {
            selectModel.addEventListener("change", (e) => {
                localStorage.setItem("setting_model_mode", e.target.value);
                showToast("Model preference saved", "success");
            });
        }

        if (sliderThresh) {
            sliderThresh.addEventListener("input", (e) => {
                if (threshVal) threshVal.innerText = e.target.value;
                localStorage.setItem("setting_threshold", e.target.value);
            });
            sliderThresh.addEventListener("change", () => {
                showToast("Threshold updated", "success");
            });
        }

        if (toggleRedirect) {
            toggleRedirect.addEventListener("change", (e) => {
                localStorage.setItem("setting_auto_redirect", e.target.checked);
                showToast("Navigation preference updated", "success");
            });
        }

        if (toggleAutoPredict) {
            toggleAutoPredict.addEventListener("change", (e) => {
                localStorage.setItem("setting_auto_predict", e.target.checked);
                showToast("Prediction trigger saved", "success");
            });
        }
    }

    // -----------------------------------------------------------------
    // 11. SQLite History Search
    // -----------------------------------------------------------------
    const searchInput = document.getElementById("historySearchInput");
    if (searchInput) {
        searchInput.addEventListener("input", (e) => {
            const query = e.target.value.toLowerCase();
            const rows = document.querySelectorAll(".history-row-item");
            rows.forEach(row => {
                const text = row.querySelector(".history-text").innerText.toLowerCase();
                const type = row.querySelector(".history-type").innerText.toLowerCase();
                if (text.includes(query) || type.includes(query)) {
                    row.style.display = "";
                } else {
                    row.style.display = "none";
                }
            });
        });
    }
});
