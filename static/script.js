const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const uploadProgressContainer = document.getElementById('uploadProgressContainer');
const uploadProgressBar = document.getElementById('uploadProgressBar');

const resultImage = document.getElementById('resultImage');
const resultVideo = document.getElementById('resultVideo');
const beeCountDiv = document.getElementById('beeCount');
const alertNotice = document.getElementById('alertNotice');

const BASE_URL = (window.location.origin && window.location.origin !== 'null' && !window.location.origin.startsWith('file'))
  ? window.location.origin
  : 'http://127.0.0.1:8000';

uploadBtn.addEventListener('click', () => {
  if (!fileInput.files.length) {
    alert("Please select a file first");
    return;
  }
  uploadFile(fileInput.files[0]);
});

function uploadFile(file) {
  resultImage.classList.add('hidden');
  resultVideo.classList.add('hidden');
  beeCountDiv.classList.add('hidden');
  alertNotice.style.display = 'none';
  beeCountDiv.textContent = '';
  alertNotice.textContent = '';

  uploadProgressContainer.style.display = 'block';
  uploadProgressBar.style.width = '0%';


  uploadBtn.disabled = true;
  uploadBtn.textContent = "Uploading...";

  const xhr = new XMLHttpRequest();
  const endpoint = file.type.startsWith('video') ? '/predict_video/' : '/predict/';
  xhr.open("POST", BASE_URL + endpoint, true);
  xhr.onerror = () => { alert("Network error while contacting server."); resetButton(); };

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const percent = (e.loaded / e.total) * 100;
      uploadProgressBar.style.width = percent + '%';
    }
  };

  xhr.onreadystatechange = () => {
    if (xhr.readyState === 4) {
      if (xhr.status !== 200) {
        let msg = 'Server error (' + xhr.status + ')';
        try { msg += ': ' + (JSON.parse(xhr.responseText).error || ''); } catch (_) { }
        alert(msg);
        resetButton();
        return;
      }
      uploadProgressBar.style.width = '100%';
      uploadBtn.textContent = "Processing...";

      handleResult(JSON.parse(xhr.responseText));
    }
  };

  const form = new FormData();
  form.append('file', file);
  xhr.send(form);
}

function handleResult(data) {
  if (data.error) {
    alert("Error: " + data.error);
    resetButton();
    return;
  }

  if (data.image_url) {
    resultImage.src = data.image_url;
    resultImage.classList.remove('hidden');
    beeCountDiv.textContent = '🐝 Detected Bees: ' + (data.count ?? '0');
    beeCountDiv.classList.remove('hidden');
  }

  if (data.video_url) {
    const src = document.getElementById('resultVideoSource');
    src.src = data.video_url;
    resultVideo.load();
    resultVideo.classList.remove('hidden');
    resultVideo.oncanplay = () => { try { resultVideo.play().catch(() => { }); } catch (_) { } };

    const frameCounts = data.frame_counts || {};
    const fps = data.fps || 20;

    beeCountDiv.classList.remove('hidden');
    function updateBeeCount() {
      const currentFrame = Math.floor(resultVideo.currentTime * fps);
      beeCountDiv.textContent = '🐝 Bees in Frame: ' + (frameCounts[currentFrame] || 0);
    }
    resultVideo.removeEventListener('timeupdate', updateBeeCount);
    resultVideo.addEventListener('timeupdate', updateBeeCount);
  }

  // Show Telegram alert notice if sent
  if (data.sms_sent) {
    alertNotice.textContent = '✅ Telegram alert sent!';
    alertNotice.style.display = 'inline-block';
  }

  resetButton();
}

function resetButton() {
  uploadBtn.disabled = false;
  uploadBtn.textContent = 'Upload & Detect';
  setTimeout(() => {
    uploadProgressContainer.style.display = 'none';

  }, 500);
}
