const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');


const resultImage = document.getElementById('resultImage');
const resultVideo = document.getElementById('resultVideo');
const beeCountDiv = document.getElementById('beeCount');
const alertNotice = document.getElementById('alertNotice');

const BASE_URL = (window.location.origin && window.location.origin.includes(':8000'))
  ? window.location.origin
  : 'http://127.0.0.1:8000';

console.log("Script loaded. BASE_URL:", BASE_URL);

uploadBtn.addEventListener('click', () => {
  console.log("Analyze button clicked");
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




  uploadBtn.disabled = true;
  uploadBtn.textContent = "Uploading...";

  const xhr = new XMLHttpRequest();
  const endpoint = file.type.startsWith('video') ? '/predict_video/' : '/predict/';
  xhr.open("POST", BASE_URL + endpoint, true);
  xhr.onerror = () => { alert("Network error while contacting server."); resetButton(); };

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      const percent = (e.loaded / e.total) * 100;
      if (percent >= 100) {
        uploadBtn.textContent = "Processing...";
      }
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

    // Show peak count initially
    const counts = Object.values(frameCounts);
    const maxCount = counts.length ? Math.max(...counts) : 0;
    beeCountDiv.textContent = '🐝 Video Analysis: Peak Bee Count = ' + maxCount;
    beeCountDiv.classList.remove('hidden');

    function updateBeeCount() {
      const currentFrame = Math.floor(resultVideo.currentTime * fps);
      // Only update if we have data for this frame
      if (frameCounts[currentFrame] !== undefined) {
        beeCountDiv.textContent = '🐝 Bees in Frame: ' + frameCounts[currentFrame];
      }
    }

    // Remove old listener if exists (best practice would be to track the function reference, 
    // but for now clearing on new upload logic handles most conflicts. 
    // We'll attach new one.)
    if (window.currentVideoListener) {
      resultVideo.removeEventListener('timeupdate', window.currentVideoListener);
    }
    window.currentVideoListener = updateBeeCount;
    resultVideo.addEventListener('timeupdate', updateBeeCount);
  }

  // Show Telegram alert notice if sent
  // Show Telegram alert notice
  alertNotice.style.display = 'block';
  alertNotice.style.marginTop = '10px';
  alertNotice.style.fontWeight = 'bold';

  if (data.sms_sent) {
    alertNotice.textContent = '✅ Telegram alert sent!';
    alertNotice.style.color = 'green';
  } else if (data.sms_error) {
    alertNotice.textContent = '❌ Telegram error: ' + data.sms_error;
    alertNotice.style.color = 'red';
  } else {
    alertNotice.textContent = 'ℹ️ No alert sent (Bee count normal)';
    alertNotice.style.color = '#0066cc';
  }

  resetButton();
}

function resetButton() {
  uploadBtn.disabled = false;
  uploadBtn.textContent = 'Analyze Now';
}
