// ai-pixel: In-browser single-neuron classifier
// Trains via gradient descent, encodes into one pixel's RGB values

(function () {
    "use strict";

    const WEIGHT_MIN = -4.0;
    const WEIGHT_MAX = 4.0;
    const WEIGHT_RANGE = WEIGHT_MAX - WEIGHT_MIN;
    const CANVAS_SIZE = 400;

    // ---- Codec ----
    function weightToByte(w) {
        const clamped = Math.max(WEIGHT_MIN, Math.min(WEIGHT_MAX, w));
        return Math.round(((clamped - WEIGHT_MIN) / WEIGHT_RANGE) * 255);
    }

    function byteToWeight(b) {
        return (b / 255) * WEIGHT_RANGE + WEIGHT_MIN;
    }

    // ---- Model ----
    function sigmoid(z) {
        z = Math.max(-500, Math.min(500, z));
        return 1 / (1 + Math.exp(-z));
    }

    function createModel() {
        return { w: [0, 0], b: 0, history: [] };
    }

    function getWeights(model) {
        return { w: model.w, b: model.b };
    }

    function forward(model, x1, x2) {
        return sigmoid(model.w[0] * x1 + model.w[1] * x2 + model.b);
    }

    function trainStep(model, X, y, lr) {
        const n = y.length;
        let dlW0 = 0, dlW1 = 0, dlB = 0;
        let loss = 0;

        for (let i = 0; i < n; i++) {
            const yHat = forward(model, X[i][0], X[i][1]);
            const yHatClip = Math.max(1e-7, Math.min(1 - 1e-7, yHat));
            loss += -(y[i] * Math.log(yHatClip) + (1 - y[i]) * Math.log(1 - yHatClip));
            const err = yHat - y[i];
            dlW0 += err * X[i][0];
            dlW1 += err * X[i][1];
            dlB += err;
        }

        loss /= n;
        dlW0 /= n;
        dlW1 /= n;
        dlB /= n;

        model.w[0] -= lr * dlW0;
        model.w[1] -= lr * dlW1;
        model.b -= lr * dlB;

        // Project back to feasible region [-4, 4]
        model.w[0] = Math.max(WEIGHT_MIN, Math.min(WEIGHT_MAX, model.w[0]));
        model.w[1] = Math.max(WEIGHT_MIN, Math.min(WEIGHT_MAX, model.w[1]));
        model.b = Math.max(WEIGHT_MIN, Math.min(WEIGHT_MAX, model.b));

        model.history.push(loss);
        return loss;
    }

    function modelToPixel(model) {
        const { w, b } = getWeights(model);
        return [weightToByte(w[0]), weightToByte(w[1]), weightToByte(b)];
    }

    function pixelToModel(r, g, b) {
        const model = createModel();
        model.w[0] = byteToWeight(r);
        model.w[1] = byteToWeight(g);
        model.b = byteToWeight(b);
        return model;
    }

    function accuracy(model, X, y) {
        let correct = 0;
        for (let i = 0; i < y.length; i++) {
            const pred = forward(model, X[i][0], X[i][1]) >= 0.5 ? 1 : 0;
            if (pred === y[i]) correct++;
        }
        return correct / y.length;
    }

    // ---- Example datasets ----
    function seededRandom(seed) {
        let s = seed;
        return function () {
            s = (s * 1103515245 + 12345) & 0x7fffffff;
            return s / 0x7fffffff;
        };
    }

    function generateCluster(rng, n, xLo, xHi, yLo, yHi) {
        const pts = [];
        for (let i = 0; i < n; i++) {
            pts.push([xLo + rng() * (xHi - xLo), yLo + rng() * (yHi - yLo)]);
        }
        return pts;
    }

    const EXAMPLES = {
        umbrella: {
            name: "Umbrella",
            features: ["Rain chance", "Wind speed"],
            classes: ["Leave it", "Bring umbrella"],
            generate() {
                const rng = seededRandom(42);
                const yes = generateCluster(rng, 30, 0.5, 1.0, 0.3, 1.0);
                const no = generateCluster(rng, 30, 0.0, 0.45, 0.0, 0.6);
                return { X: [...yes, ...no], y: [...Array(30).fill(1), ...Array(30).fill(0)] };
            }
        },
        sunscreen: {
            name: "Sunscreen",
            features: ["UV index", "Hours outside"],
            classes: ["Skip it", "Wear sunscreen"],
            generate() {
                const rng = seededRandom(43);
                const yes = generateCluster(rng, 30, 0.45, 1.0, 0.3, 1.0);
                const no = generateCluster(rng, 30, 0.0, 0.4, 0.0, 0.5);
                return { X: [...yes, ...no], y: [...Array(30).fill(1), ...Array(30).fill(0)] };
            }
        },
        escalate: {
            name: "Escalate Ticket",
            features: ["Sentiment (low=angry)", "Severity"],
            classes: ["Handle normally", "Escalate"],
            generate() {
                const rng = seededRandom(44);
                const yes = generateCluster(rng, 30, 0.0, 0.45, 0.5, 1.0);
                const no = generateCluster(rng, 30, 0.5, 1.0, 0.0, 0.5);
                return { X: [...yes, ...no], y: [...Array(30).fill(1), ...Array(30).fill(0)] };
            }
        },
        xor: {
            name: "XOR (Unsolvable)",
            features: ["X", "Y"],
            classes: ["Class 0", "Class 1"],
            generate() {
                const rng = seededRandom(45);
                const n = 20;
                const c1a = generateCluster(rng, n, 0.0, 0.4, 0.6, 1.0);
                const c1b = generateCluster(rng, n, 0.6, 1.0, 0.0, 0.4);
                const c0a = generateCluster(rng, n, 0.6, 1.0, 0.6, 1.0);
                const c0b = generateCluster(rng, n, 0.0, 0.4, 0.0, 0.4);
                return {
                    X: [...c1a, ...c1b, ...c0a, ...c0b],
                    y: [...Array(2 * n).fill(1), ...Array(2 * n).fill(0)]
                };
            }
        }
    };

    // ---- Canvas rendering ----
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const lossCanvas = document.getElementById("loss-canvas");
    const lossCtx = lossCanvas.getContext("2d");

    let points = [];
    let currentClass = 0;
    let currentModel = null;
    let currentMeta = null;

    // Tooltip element for hover predictions
    const tooltip = document.createElement("div");
    tooltip.className = "canvas-tooltip";
    tooltip.style.opacity = "0";
    document.querySelector(".canvas-wrapper").appendChild(tooltip);

    function drawCanvas() {
        const w = canvas.width;
        const h = canvas.height;
        ctx.clearRect(0, 0, w, h);

        // Confidence heatmap
        if (currentModel) {
            const res = 4;
            for (let px = 0; px < w; px += res) {
                for (let py = 0; py < h; py += res) {
                    const x1 = px / w;
                    const x2 = 1 - py / h;
                    const prob = forward(currentModel, x1, x2);
                    const r = Math.round(255 * (1 - prob) * 0.85 + 255 * 0.15);
                    const g = Math.round(78 * prob + 200 * (1 - prob) * 0.3 + 50);
                    const b = Math.round(196 * prob + 107 * (1 - prob) * 0.3 + 50);
                    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.25)`;
                    ctx.fillRect(px, py, res, res);
                }
            }

            // Decision boundary line
            const { w: weights, b: bias } = getWeights(currentModel);
            if (Math.abs(weights[1]) > 0.001) {
                ctx.beginPath();
                ctx.strokeStyle = "#e6edf3";
                ctx.lineWidth = 2;
                for (let px = 0; px <= w; px++) {
                    const x1 = px / w;
                    const x2 = -(weights[0] * x1 + bias) / weights[1];
                    const py = (1 - x2) * h;
                    if (px === 0) ctx.moveTo(px, py);
                    else ctx.lineTo(px, py);
                }
                ctx.stroke();
            }
        }

        // Grid
        ctx.strokeStyle = "rgba(48, 54, 61, 0.5)";
        ctx.lineWidth = 1;
        for (let i = 1; i < 4; i++) {
            const pos = (i / 4) * w;
            ctx.beginPath(); ctx.moveTo(pos, 0); ctx.lineTo(pos, h); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(0, pos); ctx.lineTo(w, pos); ctx.stroke();
        }

        // Data points
        points.forEach(p => {
            const px = p.x * w;
            const py = (1 - p.y) * h;
            ctx.beginPath();
            ctx.arc(px, py, 5, 0, Math.PI * 2);
            ctx.fillStyle = p.label === 0 ? "#ff6b6b" : "#4ecdc4";
            ctx.fill();
            ctx.strokeStyle = "#e6edf3";
            ctx.lineWidth = 1;
            ctx.stroke();
        });
    }

    function drawLoss(history) {
        const section = document.getElementById("loss-section");
        section.style.display = "block";

        const w = lossCanvas.width;
        const h = lossCanvas.height;
        const dpr = window.devicePixelRatio || 1;
        lossCanvas.width = lossCanvas.clientWidth * dpr;
        lossCanvas.height = lossCanvas.clientHeight * dpr;
        lossCtx.scale(dpr, dpr);
        const cw = lossCanvas.clientWidth;
        const ch = lossCanvas.clientHeight;

        lossCtx.clearRect(0, 0, cw, ch);

        if (history.length < 2) return;

        const maxLoss = Math.max(...history);
        const minLoss = Math.min(...history);
        const range = maxLoss - minLoss || 1;
        const pad = 10;

        lossCtx.beginPath();
        lossCtx.strokeStyle = "#6c5ce7";
        lossCtx.lineWidth = 1.5;

        for (let i = 0; i < history.length; i++) {
            const x = pad + (i / (history.length - 1)) * (cw - 2 * pad);
            const y = pad + (1 - (history[i] - minLoss) / range) * (ch - 2 * pad);
            if (i === 0) lossCtx.moveTo(x, y);
            else lossCtx.lineTo(x, y);
        }
        lossCtx.stroke();

        lossCtx.fillStyle = "#8b949e";
        lossCtx.font = "11px -apple-system, sans-serif";
        lossCtx.fillText(maxLoss.toFixed(3), pad, pad + 10);
        lossCtx.fillText(minLoss.toFixed(3), pad, ch - pad);
        lossCtx.fillText(`${history.length} epochs`, cw - pad - 60, ch - pad);
    }

    // ---- UI event handlers ----
    canvas.addEventListener("click", function (e) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const px = (e.clientX - rect.left) * scaleX;
        const py = (e.clientY - rect.top) * scaleY;
        const x = px / canvas.width;
        const y = 1 - py / canvas.height;

        points.push({ x, y, label: currentClass });
        document.getElementById("point-count").textContent = `${points.length} points`;
        document.getElementById("btn-train").disabled = points.length < 2;
        drawCanvas();
    });

    document.getElementById("btn-class-0").addEventListener("click", function () {
        currentClass = 0;
        this.classList.add("active");
        document.getElementById("btn-class-1").classList.remove("active");
    });

    document.getElementById("btn-class-1").addEventListener("click", function () {
        currentClass = 1;
        this.classList.add("active");
        document.getElementById("btn-class-0").classList.remove("active");
    });

    document.getElementById("btn-clear").addEventListener("click", function () {
        points = [];
        currentModel = null;
        currentMeta = null;
        document.getElementById("point-count").textContent = "0 points";
        document.getElementById("btn-train").disabled = true;
        document.getElementById("result-section").style.display = "none";
        document.getElementById("predict-section").style.display = "none";
        document.getElementById("loss-section").style.display = "none";
        document.getElementById("training-status").textContent = "";
        tooltip.style.opacity = "0";
        drawCanvas();
    });

    // ---- Hover prediction ----
    canvas.addEventListener("mousemove", function (e) {
        if (!currentModel) {
            tooltip.style.opacity = "0";
            return;
        }
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const px = (e.clientX - rect.left) * scaleX;
        const py = (e.clientY - rect.top) * scaleY;
        const x1 = px / canvas.width;
        const x2 = 1 - py / canvas.height;
        const prob = forward(currentModel, x1, x2);
        const cls = prob >= 0.5 ? 1 : 0;

        const classNames = currentMeta ? currentMeta.classes : ["Class 0", "Class 1"];
        const label = classNames[cls];
        const color = cls === 1 ? "#4ecdc4" : "#ff6b6b";

        tooltip.innerHTML = `<span style="color:${color}">${label}</span> ${(prob * 100).toFixed(1)}%`;
        tooltip.style.opacity = "1";
        tooltip.style.left = (e.clientX - rect.left + 12) + "px";
        tooltip.style.top = (e.clientY - rect.top - 10) + "px";

        const readout = document.getElementById("hover-readout");
        if (readout) {
            const f1 = currentMeta ? currentMeta.features[0] : "Feature 1";
            const f2 = currentMeta ? currentMeta.features[1] : "Feature 2";
            readout.innerHTML = `${f1}=${x1.toFixed(2)}, ${f2}=${x2.toFixed(2)} → <span style="color:${color}">${label}</span> (${(prob * 100).toFixed(1)}%)`;
            readout.classList.add("active");
        }
    });

    canvas.addEventListener("mouseleave", function () {
        tooltip.style.opacity = "0";
        const readout = document.getElementById("hover-readout");
        if (readout && currentModel) {
            readout.textContent = "Hover over the canvas to see live predictions";
            readout.classList.remove("active");
        }
    });

    // ---- Training with animation ----
    document.getElementById("btn-train").addEventListener("click", function () {
        const X = points.map(p => [p.x, p.y]);
        const y = points.map(p => p.label);
        const epochs = parseInt(document.getElementById("epochs").value) || 500;
        const lr = parseFloat(document.getElementById("lr").value) || 0.2;

        const model = createModel();
        currentModel = model;

        const statusEl = document.getElementById("training-status");
        const btn = this;
        btn.disabled = true;

        let epoch = 0;
        const step = Math.max(1, Math.floor(epochs / 100));

        function animate() {
            for (let i = 0; i < step && epoch < epochs; i++, epoch++) {
                trainStep(model, X, y, lr);
            }

            currentModel = model;
            drawCanvas();

            const acc = accuracy(model, X, y);
            statusEl.textContent = `Epoch ${epoch}/${epochs} — accuracy: ${(acc * 100).toFixed(1)}%`;

            if (epoch < epochs) {
                requestAnimationFrame(animate);
            } else {
                btn.disabled = false;
                showResult(model, X, y);
            }
        }

        requestAnimationFrame(animate);
    });

    function showResult(model, X, y) {
        const pixel = modelToPixel(model);
        const { w, b } = getWeights(model);

        const section = document.getElementById("result-section");
        section.style.display = "block";

        document.getElementById("pixel-swatch").style.backgroundColor =
            `rgb(${pixel[0]}, ${pixel[1]}, ${pixel[2]})`;

        document.getElementById("pixel-rgb").textContent =
            `RGB(${pixel[0]}, ${pixel[1]}, ${pixel[2]})`;

        const hex = "#" + pixel.map(v => v.toString(16).padStart(2, "0")).join("");
        document.getElementById("pixel-hex").textContent = hex;

        document.getElementById("pixel-weights").textContent =
            `W=[${w[0].toFixed(3)}, ${w[1].toFixed(3)}] B=${b.toFixed(3)}`;

        if (y.length > 0) {
            const acc = accuracy(model, X, y);
            document.getElementById("pixel-accuracy").textContent =
                `Accuracy: ${(acc * 100).toFixed(1)}%`;
        } else {
            document.getElementById("pixel-accuracy").textContent =
                "Add data points to test accuracy";
        }

        drawLoss(model.history);

        // Show predict section with correct labels
        const predictSection = document.getElementById("predict-section");
        predictSection.style.display = "block";
        const f1 = currentMeta ? currentMeta.features[0] : "Feature 1";
        const f2 = currentMeta ? currentMeta.features[1] : "Feature 2";
        document.getElementById("predict-label-1").textContent = f1;
        document.getElementById("predict-label-2").textContent = f2;
        document.getElementById("predict-result").innerHTML = "";
    }

    // ---- Download ----
    document.getElementById("btn-download").addEventListener("click", function () {
        if (!currentModel) return;
        const pixel = modelToPixel(currentModel);
        const c = document.createElement("canvas");
        c.width = 1; c.height = 1;
        const cx = c.getContext("2d");
        const id = cx.createImageData(1, 1);
        id.data[0] = pixel[0];
        id.data[1] = pixel[1];
        id.data[2] = pixel[2];
        id.data[3] = 255;
        cx.putImageData(id, 0, 0);
        c.toBlob(function (blob) {
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "ai-pixel-model.png";
            a.click();
            URL.revokeObjectURL(url);
        }, "image/png");
    });

    // ---- Manual prediction ----
    document.getElementById("btn-predict").addEventListener("click", function () {
        if (!currentModel) return;
        const x1 = parseFloat(document.getElementById("predict-x1").value);
        const x2 = parseFloat(document.getElementById("predict-x2").value);
        if (isNaN(x1) || isNaN(x2)) return;

        const prob = forward(currentModel, x1, x2);
        const cls = prob >= 0.5 ? 1 : 0;
        const classNames = currentMeta ? currentMeta.classes : ["Class 0", "Class 1"];
        const label = classNames[cls];
        const colorClass = cls === 1 ? "class-1-text" : "class-0-text";

        document.getElementById("predict-result").innerHTML =
            `<span class="${colorClass}">${label}</span> — <span class="prob">${(prob * 100).toFixed(1)}%</span> confidence`;
    });

    // ---- File upload / drag-drop ----
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");

    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", function (e) {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        const file = e.dataTransfer.files[0];
        if (file) loadPixelFile(file);
    });
    fileInput.addEventListener("change", function () {
        if (this.files[0]) loadPixelFile(this.files[0]);
    });

    function loadPixelFile(file) {
        const img = new Image();
        img.onload = function () {
            const c = document.createElement("canvas");
            c.width = 1; c.height = 1;
            const cx = c.getContext("2d");
            cx.drawImage(img, 0, 0, 1, 1);
            const data = cx.getImageData(0, 0, 1, 1).data;
            const model = pixelToModel(data[0], data[1], data[2]);
            currentModel = model;
            drawCanvas();
            showResult(model, points.map(p => [p.x, p.y]), points.map(p => p.label));
            document.getElementById("training-status").textContent = "Loaded model from pixel";
        };
        img.src = URL.createObjectURL(file);
    }

    // ---- Example datasets ----
    document.querySelectorAll(".btn-example").forEach(btn => {
        btn.addEventListener("click", function () {
            const key = this.dataset.example;
            const ex = EXAMPLES[key];
            if (!ex) return;

            const data = ex.generate();
            points = data.X.map((coords, i) => ({
                x: coords[0], y: coords[1], label: data.y[i]
            }));
            currentModel = null;
            currentMeta = { features: ex.features, classes: ex.classes };
            document.getElementById("point-count").textContent = `${points.length} points`;
            document.getElementById("btn-train").disabled = false;
            document.getElementById("result-section").style.display = "none";
            document.getElementById("predict-section").style.display = "none";
            document.getElementById("loss-section").style.display = "none";
            document.getElementById("training-status").textContent = `Loaded: ${ex.name}`;

            // Update axis labels
            document.querySelector(".axis-x").textContent = ex.features[0];
            document.querySelector(".axis-y").textContent = ex.features[1];

            drawCanvas();
        });
    });

    // ---- Init ----
    drawCanvas();
})();
