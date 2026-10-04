const fileInput = document.getElementById('fileInput');
const cameraInput = document.getElementById('cameraInput');
const cameraBtn = document.getElementById('cameraBtn');
const dropZone = document.getElementById('dropZone');
const previewPanel = document.getElementById('previewPanel');
const resultCard = document.getElementById('resultCard');
const previewImage = document.getElementById('previewImage');
const resultTitle = document.getElementById('resultTitle');
const resultText = document.getElementById('resultText');
const progressBar = document.getElementById('progressBar');
const confidenceValue = document.getElementById('confidenceValue');
const newImageBtn = document.getElementById('newImageBtn');

async function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;

  const reader = new FileReader();
  reader.onload = e => {
    previewPanel.innerHTML = `<img src="${e.target.result}" alt="Selected plant leaf" style="display:block">`;
    previewImage.src = e.target.result;
    resultCard.hidden = false;
    resultTitle.textContent = 'Analyzing crop leaf...';
    resultText.textContent = 'Sending the image to the PlantCare AI model.';
    confidenceValue.textContent = '...';
    progressBar.style.width = '15%';
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };
  reader.readAsDataURL(file);

  const form = new FormData();
  form.append('image', file);

  try {
    const response = await fetch('/api/predict', { method: 'POST', body: form });
    const data = await response.json();

    if (data.status === 'model_not_ready') {
      resultTitle.textContent = 'AI model not installed yet';
      resultText.textContent = data.message;
      confidenceValue.textContent = 'Pending';
      progressBar.style.width = '0%';
      return;
    }

    if (data.status === 'low_confidence') {
      resultTitle.textContent = 'Unclear prediction';
      resultText.textContent = data.message;
      confidenceValue.textContent = 'Low';
      progressBar.style.width = '25%';
      return;
    }

    if (data.status === 'ok') {
      const pct = (data.confidence * 100).toFixed(1);
      resultTitle.textContent = `${data.plant} — ${data.disease}`;
      resultText.textContent = 'The model classified the uploaded crop leaf using the trained disease classes.';
      confidenceValue.textContent = `${pct}%`;
      progressBar.style.width = `${Math.max(3, Math.min(100, data.confidence * 100))}%`;
    } else {
      throw new Error(data.error || 'Prediction failed.');
    }
  } catch (err) {
    resultTitle.textContent = 'Prediction unavailable';
    resultText.textContent = err.message;
    confidenceValue.textContent = '—';
    progressBar.style.width = '0%';
  }
}

fileInput.addEventListener('change', e => handleFile(e.target.files[0]));
cameraBtn.addEventListener('click', () => cameraInput.click());
cameraInput.addEventListener('change', e => handleFile(e.target.files[0]));

['dragenter', 'dragover'].forEach(event => dropZone.addEventListener(event, e => {
  e.preventDefault();
  dropZone.classList.add('dragging');
}));

['dragleave', 'drop'].forEach(event => dropZone.addEventListener(event, e => {
  e.preventDefault();
  dropZone.classList.remove('dragging');
}));

dropZone.addEventListener('drop', e => handleFile(e.dataTransfer.files[0]));

newImageBtn.addEventListener('click', () => {
  resultCard.hidden = true;
  fileInput.value = '';
  cameraInput.value = '';
  previewPanel.innerHTML = `<div class="preview-placeholder"><span>🌱</span><p>Your selected image will appear here.</p></div>`;
  document.getElementById('detect').scrollIntoView({ behavior: 'smooth' });
});
