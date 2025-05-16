const API_BASE = 'http://0.0.0.0:8002';

document.getElementById('generate').addEventListener('click', async () => {
  const prompt = document.getElementById('prompt').value;
  if (!prompt) return;
  const resultsDiv = document.getElementById('results');
  resultsDiv.innerHTML = '';
  try {
    const res = await fetch(`${API_BASE}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt })
    });
    const data = await res.json();
    data.results.forEach(item => {
      if (item.type === 'text') {
        const p = document.createElement('p');
        p.textContent = item.data;
        resultsDiv.appendChild(p);
      } else if (item.type === 'image') {
        const img = document.createElement('img');
        img.src = item.data;
        resultsDiv.appendChild(img);
      }
    });
  } catch (err) {
    console.error(err);
    resultsDiv.textContent = 'Error generating content';
  }
});

document.getElementById('generateWithImage').addEventListener('click', () => {
  const fileInput = document.getElementById('inputImage');
  const desc = document.getElementById('desc').value;
  const resultsDiv = document.getElementById('resultsImg');
  resultsDiv.innerHTML = '';

  if (!fileInput.files.length) return;
  const file = fileInput.files[0];
  const reader = new FileReader();
  reader.onload = async (e) => {
    const base64 = e.target.result;
    try {
      const res = await fetch(`${API_BASE}/generate_with_image`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: desc, image: base64 })
      });
      const data = await res.json();
      data.results.forEach(item => {
        if (item.type === 'text') {
          const p = document.createElement('p');
          p.textContent = item.data;
          resultsDiv.appendChild(p);
        } else if (item.type === 'image') {
          const img = document.createElement('img');
          img.src = item.data;
          resultsDiv.appendChild(img);
        }
      });
    } catch (err) {
      console.error(err);
      resultsDiv.textContent = 'Error generating content';
    }
  };
  reader.readAsDataURL(file);
});
