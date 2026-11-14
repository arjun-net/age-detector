const fileInput = document.querySelector("#file");
const pasteButton = document.querySelector("#paste");
const cameraButton = document.querySelector("#camera");
const captureButton = document.querySelector("#capture");
const estimateButton = document.querySelector("#estimate");
const video = document.querySelector("#video");
const preview = document.querySelector("#preview");
const placeholder = document.querySelector("#placeholder");
const results = document.querySelector("#results");
const resultContent = document.querySelector("#result-content");

let mediaStream = null;
let selectedBlob = null;

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  stopCamera();
  setSelectedImage(file);
});

pasteButton.addEventListener("click", async () => {
  try {
    const clipboardItems = await navigator.clipboard.read();
    const imageItem = clipboardItems.find((item) =>
      item.types.some((type) => type.startsWith("image/")),
    );
    const imageType = imageItem?.types.find((type) => type.startsWith("image/"));
    const blob = imageType ? await imageItem.getType(imageType) : null;

    if (!blob) {
      showError("The clipboard does not contain an image. Copy an image and try again.");
      return;
    }

    stopCamera();
    setSelectedImage(blob);
  } catch (error) {
    showError("Automatic clipboard reading is unavailable. Press Ctrl+V (or Command+V) to paste the copied image.");
  }
});

document.addEventListener("paste", (event) => {
  const imageFile = [...(event.clipboardData?.files ?? [])].find((file) =>
    file.type.startsWith("image/"),
  );
  if (!imageFile) return;

  event.preventDefault();
  stopCamera();
  setSelectedImage(imageFile);
});

cameraButton.addEventListener("click", async () => {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = mediaStream;
    video.hidden = false;
    preview.hidden = true;
    placeholder.hidden = true;
    cameraButton.hidden = true;
    captureButton.hidden = false;
  } catch (error) {
    resultContent.innerHTML = '<p class="error">Camera access was denied or is unavailable.</p>';
    results.hidden = false;
  }
});

captureButton.addEventListener("click", () => {
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);

  canvas.toBlob(
    (blob) => {
      if (!blob) return;
      setSelectedImage(blob);
      video.hidden = true;
      stopCamera();
    },
    "image/jpeg",
    0.92,
  );
});

estimateButton.addEventListener("click", async () => {
  if (!selectedBlob) return;

  estimateButton.disabled = true;
  resultContent.innerHTML = '<p class="loading">Detecting faces and estimating age…</p>';
  results.hidden = false;

  const formData = new FormData();
  formData.append("image", selectedBlob, "capture.jpg");

  try {
    const response = await fetch("/api/estimate", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The estimate could not be completed.");
    renderResults(data);
  } catch (error) {
    resultContent.innerHTML = `<p class="error">${escapeHtml(error.message)}</p>`;
  } finally {
    estimateButton.disabled = false;
  }
});

function setSelectedImage(blob) {
  selectedBlob = blob;
  preview.src = URL.createObjectURL(blob);
  preview.hidden = false;
  placeholder.hidden = true;
  estimateButton.disabled = false;
  results.hidden = true;
}

function showError(message) {
  resultContent.innerHTML = `<p class="error">${escapeHtml(message)}</p>`;
  results.hidden = false;
}

function renderResults(data) {
  if (data.count === 0) {
    resultContent.innerHTML = "<p>No faces were detected. Try a clear, front-facing photo in good lighting.</p>";
    return;
  }

  const cards = data.faces
    .map(
      (face) => `
        <div class="result-card">
          <p class="result-age">${face.age} years</p>
          <p class="result-label">Likely age band: ${escapeHtml(face.age_range)} years</p>
          <p class="meta">Face detection confidence: ${Math.round(face.detection_confidence * 100)}%</p>
        </div>`,
    )
    .join("");

  resultContent.innerHTML = `<div class="faces">${cards}</div><p class="meta">${escapeHtml(data.notice)}</p>`;
}

function stopCamera() {
  mediaStream?.getTracks().forEach((track) => track.stop());
  mediaStream = null;
  video.srcObject = null;
  captureButton.hidden = true;
  cameraButton.hidden = false;
}

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = value;
  return element.innerHTML;
}
