// ============================================
//  BeeVision — Frontend Script
// ============================================

// FastAPI serves both the frontend AND the API from the same origin,
// so window.location.origin is always the correct base URL —
// whether running locally (http://127.0.0.1:8000) or deployed.
const BASE_URL = window.location.origin;

// ─── Page Navigation ─────────────────────────
function showPage(pageId) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page-' + pageId).classList.add('active');

  // Update nav active styles
  document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
  const navEl = document.getElementById('nav-' + pageId);
  if (navEl) navEl.classList.add('active');

  window.scrollTo({ top: 0, behavior: 'smooth' });

  // Fetch device info whenever detection page is shown
  if (pageId === 'detection') fetchDeviceInfo();
  return false;
}

// ─── Device Info (GPU / CPU status) ──────────
function fetchDeviceInfo() {
  const bar = document.getElementById('deviceStatusBar');
  const dot = document.getElementById('deviceStatusDot');
  const label = document.getElementById('deviceStatusLabel');
  const sub = document.getElementById('deviceStatusSub');
  const detail = document.getElementById('deviceStatusDetails');
  if (!bar) return;

  // Reset to loading state
  bar.className = 'device-status-bar';
  dot.className = 'device-status-dot';
  label.className = 'device-status-label';
  if (sub) { sub.className = 'device-status-sub'; sub.textContent = ''; }
  detail.className = 'device-status-details';
  label.textContent = 'Detecting hardware...';
  detail.textContent = '';

  fetch(BASE_URL + '/device_info')
    .then(r => r.json())
    .then(data => {
      if (data.gpu_available) {
        // ── GPU path ───────────────────────────────────────────────
        const gpuName = data.gpu_name || 'CUDA GPU';
        const vram = data.vram_gb ? ` (${data.vram_gb} GB VRAM)` : '';

        bar.classList.add('gpu-ready');
        dot.classList.add('gpu');

        label.classList.add('gpu');
        label.textContent = `GPU Detected: ${gpuName}${vram}`;

        if (sub) {
          sub.classList.add('gpu');
          sub.textContent = 'Using GPU for inference — CUDA acceleration active';
        }

        detail.classList.add('gpu');
        const dLines = [];
        if (data.vram_gb) dLines.push(`VRAM: ${data.vram_gb} GB`);
        if (data.ram_gb) dLines.push(`RAM: ${data.ram_gb} GB`);
        detail.innerHTML = dLines.join('<br>');

      } else {
        // ── CPU path ───────────────────────────────────────────────
        bar.classList.add('cpu-only');
        dot.classList.add('cpu');

        label.classList.add('cpu');
        label.textContent = 'No GPU Detected — Using CPU for inference';

        if (sub) {
          sub.classList.add('cpu');
          const cpuLine = data.cpu_name
            ? `CPU: ${data.cpu_name}`
            : 'Processor details unavailable';
          sub.textContent = cpuLine;
        }

        detail.classList.add('cpu');
        const dLines = [];
        if (data.cpu_cores) dLines.push(`Cores: ${data.cpu_cores} physical`);
        if (data.cpu_threads) dLines.push(`Threads: ${data.cpu_threads}`);
        if (data.ram_gb) dLines.push(`RAM: ${data.ram_gb} GB`);
        detail.innerHTML = dLines.join('<br>');
      }
    })
    .catch(() => {
      label.textContent = 'Hardware info unavailable';
    });
}

// Fetch on initial page load if starting on detection
if (document.getElementById('page-detection') &&
  document.getElementById('page-detection').classList.contains('active')) {
  fetchDeviceInfo();
}

// ─── Navbar scroll shadow ─────────────────────
window.addEventListener('scroll', () => {
  document.getElementById('navbar').classList.toggle('scrolled', window.scrollY > 10);
});

// ─── Mobile menu toggle ───────────────────────
function toggleMobileMenu() {
  document.getElementById('navMobileMenu').classList.toggle('open');
}

// ─── Upload / Detection Logic ─────────────────
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const btnText = document.getElementById('btnText');
const fileInfo = document.getElementById('fileInfo');
const progressWrap = document.getElementById('progressWrap');
const progressBar = document.getElementById('progressBar');
const progressLabel = document.getElementById('progressLabel');

const resultsArea = document.getElementById('resultsArea');
const resultImage = document.getElementById('resultImage');
const resultVideo = document.getElementById('resultVideo');
const beeCountDiv = document.getElementById('beeCount');
const alertNotice = document.getElementById('alertNotice');

const dropzone = document.getElementById('dropzone');

// ── File selection via click ──
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) {
    const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
    fileInfo.textContent = `📄 ${file.name} (${sizeMB} MB)`;
    uploadBtn.disabled = false;
    btnText.textContent = 'Analyze Now';
  } else {
    fileInfo.textContent = '';
    uploadBtn.disabled = true;
    btnText.textContent = 'Select a file to analyze';
  }
});

// ── Drag & drop ──
dropzone.addEventListener('dragover', e => {
  e.preventDefault();
  dropzone.classList.add('drag-over');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('drag-over');
});

dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) {
    // Assign to the file input (for consistency)
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    fileInput.dispatchEvent(new Event('change'));
  }
});

// ── Analyze button ──
uploadBtn.addEventListener('click', () => {
  if (!fileInput.files.length) return;
  uploadFile(fileInput.files[0]);
});

function uploadFile(file) {
  // Reset results
  resultImage.classList.add('hidden');
  resultVideo.classList.add('hidden');
  beeCountDiv.textContent = '';
  alertNotice.style.display = 'none';
  alertNotice.innerHTML = '';
  resultsArea.style.display = 'none';

  // Disable button & show progress
  uploadBtn.disabled = true;
  btnText.textContent = 'Uploading...';
  progressWrap.style.display = 'block';
  progressBar.style.width = '0%';
  progressLabel.textContent = 'Uploading...';

  const xhr = new XMLHttpRequest();
  const endpoint = file.type.startsWith('video') ? '/predict_video/' : '/predict/';
  xhr.open('POST', BASE_URL + endpoint, true);

  xhr.onerror = () => {
    alert('Network error while contacting server.');
    resetButton();
  };

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const pct = Math.round((e.loaded / e.total) * 100);
      progressBar.style.width = pct + '%';
      if (pct >= 100) {
        progressLabel.textContent = 'Processing with AI… this may take a moment';
        // animate indeterminate
        progressBar.style.width = '80%';
      } else {
        progressLabel.textContent = `Uploading… ${pct}%`;
      }
    }
  };

  xhr.onreadystatechange = () => {
    if (xhr.readyState === 4) {
      progressBar.style.width = '100%';
      if (xhr.status !== 200) {
        let msg = 'Server error (' + xhr.status + ')';
        try { msg += ': ' + (JSON.parse(xhr.responseText).error || ''); } catch (_) { }
        alert(msg);
        resetButton();
        return;
      }
      handleResult(JSON.parse(xhr.responseText));
    }
  };

  const threshold = parseInt(document.getElementById('thresholdInput').value) || 20;
  const form = new FormData();
  form.append('file', file);
  form.append('threshold', threshold);
  xhr.send(form);
}

function handleResult(data) {
  progressWrap.style.display = 'none';

  if (data.error) {
    alert('Error: ' + data.error);
    resetButton();
    return;
  }

  resultsArea.style.display = 'block';

  if (data.image_url) {
    resultImage.src = data.image_url;
    resultImage.classList.remove('hidden');
    beeCountDiv.innerHTML =
      '🐝 Detected Bees: <strong>' + (data.count ?? '0') + '</strong>' +
      (data.infer_ms != null ? _inferBadge(data.infer_ms, data.infer_device) : '');
  }

  if (data.video_url) {
    const src = document.getElementById('resultVideoSource');
    src.src = data.video_url;
    resultVideo.load();
    resultVideo.classList.remove('hidden');
    resultVideo.oncanplay = () => { try { resultVideo.play().catch(() => { }); } catch (_) { } };

    const frameCounts = data.frame_counts || {};
    const fps = data.fps || 20;
    const counts = Object.values(frameCounts);
    const maxCount = counts.length ? Math.max(...counts) : 0;
    beeCountDiv.innerHTML =
      '🐝 Video Analysis: Peak Bee Count = <strong>' + maxCount + '</strong>' +
      (data.avg_infer_ms != null ? _inferBadge(data.avg_infer_ms, data.infer_device, true) : '');

    function updateBeeCount() {
      const currentFrame = Math.floor(resultVideo.currentTime * fps);
      if (frameCounts[currentFrame] !== undefined) {
        beeCountDiv.innerHTML =
          '🐝 Bees in Frame: <strong>' + frameCounts[currentFrame] + '</strong>' +
          (data.avg_infer_ms != null ? _inferBadge(data.avg_infer_ms, data.infer_device, true) : '');
      }
    }

    if (window.currentVideoListener) {
      resultVideo.removeEventListener('timeupdate', window.currentVideoListener);
    }
    window.currentVideoListener = updateBeeCount;
    resultVideo.addEventListener('timeupdate', updateBeeCount);
  }

  // ── Telegram alert notice with reason ─────────────────────────────────────
  alertNotice.style.display = 'block';

  const thr = data.threshold ?? '—';

  if (data.sms_error) {
    // Send was attempted but failed
    alertNotice.innerHTML =
      '<span class="alert-title">Telegram Error</span><br>' +
      'Could not send alert: ' + data.sms_error;
    alertNotice.style.background = '#fee2e2';
    alertNotice.style.color = '#991b1b';
    alertNotice.style.borderColor = '#fca5a5';

  } else if (data.sms_sent) {
    // Alert was sent — show WHY
    let reason = '';
    if (data.image_url) {
      // Image alert
      const cnt = data.count ?? '—';
      const diff = (typeof data.count === 'number' && typeof data.threshold === 'number')
        ? (data.threshold - data.count) : null;
      reason =
        `Bee count <strong>${cnt}</strong> dropped below the alert threshold of <strong>${thr}</strong>.` +
        (diff !== null ? ` That is <strong>${diff}</strong> bee${diff !== 1 ? 's' : ''} below the limit.` : '');
    } else if (data.video_url) {
      // Video alert
      const counts = Object.values(data.frame_counts || {});
      const minCount = counts.length ? Math.min(...counts) : '—';
      const diff = (typeof minCount === 'number' && typeof data.threshold === 'number')
        ? (data.threshold - minCount) : null;
      reason =
        `Minimum bee count during video dropped to <strong>${minCount}</strong>, ` +
        `below the alert threshold of <strong>${thr}</strong>.` +
        (diff !== null ? ` That is <strong>${diff}</strong> bee${diff !== 1 ? 's' : ''} below the limit.` : '');
    }
    alertNotice.innerHTML =
      '<span class="alert-title">Telegram Alert Sent</span><br>' + reason;
    alertNotice.style.background = '#d1fae5';
    alertNotice.style.color = '#065f46';
    alertNotice.style.borderColor = '#6ee7b7';

  } else {
    // No alert — show WHY (healthy count)
    let reason = '';
    if (data.image_url) {
      const cnt = data.count ?? '—';
      const surplus = (typeof data.count === 'number' && typeof data.threshold === 'number')
        ? (data.count - data.threshold) : null;
      reason =
        `Bee count <strong>${cnt}</strong> is above the threshold of <strong>${thr}</strong>.` +
        (surplus !== null ? ` Colony activity looks healthy (+${surplus} bees above limit).` : '');
    } else if (data.video_url) {
      const counts = Object.values(data.frame_counts || {});
      const minCount = counts.length ? Math.min(...counts) : '—';
      reason =
        `Bee count stayed at or above the threshold of <strong>${thr}</strong> throughout the video ` +
        `(lowest frame count: <strong>${minCount}</strong>). Colony activity looks healthy.`;
    }
    alertNotice.innerHTML =
      '<span class="alert-title">No Alert Sent</span><br>' + reason;
    alertNotice.style.background = '#eff6ff';
    alertNotice.style.color = '#1e40af';
    alertNotice.style.borderColor = '#bfdbfe';
  }

  resetButton();
}

// Build an inference time + device badge HTML string
function _inferBadge(ms, device, isAvg) {
  const isGPU = device && device.toLowerCase() !== 'cpu';
  const color = isGPU ? '#16a34a' : '#b45309';
  const bg = isGPU ? '#dcfce7' : '#fef3c7';
  const label = isGPU ? 'GPU' : 'CPU';
  const prefix = isAvg ? 'avg ' : '';
  return ` <span style="
    display:inline-block;
    margin-left:10px;
    font-size:0.78rem;
    font-weight:700;
    background:${bg};
    color:${color};
    border-radius:999px;
    padding:3px 10px;
    letter-spacing:0.3px;
    vertical-align:middle;
  ">${label} · ${prefix}${ms} ms</span>`;
}

function resetButton() {
  uploadBtn.disabled = fileInput.files.length === 0;
  btnText.textContent = fileInput.files.length ? 'Analyze Again' : 'Select a file to analyze';
  progressWrap.style.display = 'none';
}

// ─── Threshold +/− buttons ────────────────────
function adjustThreshold(delta) {
  const input = document.getElementById('thresholdInput');
  const val = Math.min(999, Math.max(1, (parseInt(input.value) || 20) + delta));
  input.value = val;
}
