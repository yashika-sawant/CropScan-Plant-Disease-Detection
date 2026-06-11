/* ═══════════════════════════════════════════════
   CropScan — script.js
   Handles: dark mode, image preview, drag & drop,
            loading overlay, history panel
═══════════════════════════════════════════════ */

// ── Dark Mode ─────────────────────────────────
function toggleDark() {
  document.body.classList.toggle('dark');
  const isDark = document.body.classList.contains('dark');
  localStorage.setItem('cropscan-dark', isDark);
  document.querySelector('.dark-toggle').textContent = isDark ? '☀️' : '🌙';
}

// Apply saved preference on load
(function () {
  if (localStorage.getItem('cropscan-dark') === 'true') {
    document.body.classList.add('dark');
    const btn = document.querySelector('.dark-toggle');
    if (btn) btn.textContent = '☀️';
  }
})();

// ── Image Preview & Upload ────────────────────
const fileInput   = document.getElementById('fileInput');
const dropZone    = document.getElementById('dropZone');
const preview     = document.getElementById('preview');
const placeholder = document.getElementById('dropPlaceholder');
const detectBtn   = document.getElementById('detectBtn');
const uploadForm  = document.getElementById('uploadForm');

function showPreview(file) {
  if (!file || !file.type.match('image.*')) return;
  const reader = new FileReader();
  reader.onload = e => {
    preview.src = e.target.result;
    preview.style.display = 'block';
    if (placeholder) placeholder.style.display = 'none';
    if (detectBtn) detectBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

if (fileInput) {
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) showPreview(fileInput.files[0]);
  });
}

// Drag & Drop
if (dropZone) {
  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) {
      // Assign to the real input so Flask receives it
      const dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
      showPreview(file);
    }
  });
}

// Show loading overlay on form submit
if (uploadForm) {
  uploadForm.addEventListener('submit', () => {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) overlay.classList.add('active');
  });
}

// ── Confidence Bar Animation (result page) ────
window.addEventListener('load', () => {
  const bar = document.querySelector('.confidence-bar-fill');
  if (bar) {
    const target = bar.style.width;
    bar.style.width = '0';
    setTimeout(() => { bar.style.width = target; }, 100);
  }
});

// ── History Panel ─────────────────────────────
async function loadHistory() {
  try {
    const res  = await fetch('/history');
    const data = await res.json();
    const section = document.getElementById('historySection');
    const grid    = document.getElementById('historyGrid');
    if (!section || !grid) return;
    if (!data.length) { section.style.display = 'none'; return; }
    section.style.display = 'block';
    grid.innerHTML = data.map(item => `
      <div class="history-item">
        <img src="/static/${item.image}" alt="${item.disease}" />
        <div class="history-item-info">
          <strong style="color:${item.color}">${item.disease}</strong>
          <span>${item.confidence}% confidence</span>
        </div>
      </div>
    `).join('');
  } catch (e) { /* silently ignore */ }
}

async function clearHistory() {
  await fetch('/clear-history', { method: 'POST' });
  const section = document.getElementById('historySection');
  if (section) section.style.display = 'none';
}

// Load history on home page
if (document.getElementById('historySection')) loadHistory();
