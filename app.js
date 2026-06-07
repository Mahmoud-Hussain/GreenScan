// GreenScan 2.0 — app.js
const API = 'http://localhost:8000';
let ws, lossChart, accChart;
let predFile = null;

// ── Page routing ──────────────────────────────────────────────────────────────
function showPage(id) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  const items = document.querySelectorAll('.nav-item');
  const labels = ['dashboard','dataset','train','viewer','fl','models','predict','compare'];
  const idx = labels.indexOf(id);
  if (idx >= 0) items[idx].classList.add('active');
  if (id === 'dashboard') loadDashboard();
  if (id === 'dataset') { loadDatasetStats(); loadThumbnails(); }
  if (id === 'train') loadTrainHistory();
  if (id === 'fl') loadFLStatus();
  if (id === 'models') loadModels();
  if (id === 'compare') loadCompare();
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectWS() {
  ws = new WebSocket('ws://localhost:8000/ws/training');
  ws.onopen = () => { document.getElementById('ws-status').innerHTML = '<span class="live-dot"></span>Connected'; };
  ws.onclose = () => { document.getElementById('ws-status').textContent = '⚪ Disconnected'; setTimeout(connectWS, 3000); };
  ws.onerror = () => { document.getElementById('ws-status').textContent = '🔴 WS Error'; };
  ws.onmessage = e => {
    try { handleWS(JSON.parse(e.data)); } catch {}
  };
}

function handleWS(msg) {
  if (msg.type === 'start') {
    document.getElementById('train-progress-card').style.display = '';
    initCharts();
    log('Training started — Device: ' + msg.device);
  }
  if (msg.type === 'epoch') {
    document.getElementById('t-epoch').textContent = msg.epoch + '/' + msg.total_epochs;
    document.getElementById('t-train-acc').textContent = msg.train_acc + '%';
    document.getElementById('t-val-acc').textContent = msg.val_acc + '%';
    document.getElementById('t-co2').textContent = msg.co2_kg.toFixed(6);
    document.getElementById('train-phase-badge').textContent = msg.phase;
    addChartPoint(msg.epoch, msg.train_loss, msg.val_loss, msg.train_acc, msg.val_acc);
    addEpochRow(msg);
  }
  if (msg.type === 'saved') log('✅ Model saved as ' + msg.version + ' — Val Acc: ' + msg.val_acc + '%');
  if (msg.type === 'done') {
    log('🏁 Training done — Best Val Acc: ' + msg.best_val_acc + '% | CO₂: ' + msg.total_co2_kg + ' kg');
    document.getElementById('btn-start-train').disabled = false;
  }
  if (msg.type === 'error') log('❌ ' + msg.message);
  if (msg.type === 'info') log('ℹ ' + msg.message);
  if (msg.type === 'fl_round') {
    addFLRoundRow(msg);
    const pct = msg.total_rounds > 0 ? Math.round((msg.round / msg.total_rounds) * 100) : 0;
    document.getElementById('fl-pbar').style.width = pct + '%';
    document.getElementById('fl-round-label').textContent = msg.round + ' / ' + msg.total_rounds;
    for (let i = 0; i < 3; i++) {
      const el = document.getElementById('fl-n' + i + '-status');
      if (el && msg.round > 0) { el.textContent = 'Round ' + msg.round; el.className = 'badge badge-green'; }
    }
  }
}

function log(msg) {
  const box = document.querySelector('.log-box');
  if (!box) return;
  box.innerHTML += `<div>[${new Date().toLocaleTimeString()}] ${msg}</div>`;
  box.scrollTop = box.scrollHeight;
}

// ── Dashboard ─────────────────────────────────────────────────────────────────
async function loadDashboard() {
  const [stats, status] = await Promise.all([
    fetch(API + '/api/dataset/stats').then(r => r.json()).catch(() => null),
    fetch(API + '/api/train/status').then(r => r.json()).catch(() => null),
  ]);
  if (stats) {
    document.getElementById('ds-total').textContent = stats.total;
    document.getElementById('ds-train').textContent = stats.splits?.train?.total || 0;
    document.getElementById('ds-val').textContent = stats.splits?.validation?.total || 0;
    document.getElementById('ds-test').textContent = stats.splits?.test?.total || 0;
    const bars = document.getElementById('dash-class-bars');
    bars.innerHTML = '';
    const trainPer = stats.splits?.train?.per_class || {};
    const max = Math.max(1, ...Object.values(trainPer));
    for (const [cls, cnt] of Object.entries(trainPer)) {
      bars.innerHTML += `<div class="cls-bar">
        <div class="name">${cls}</div>
        <div class="bar-bg"><div class="bar-fill" style="width:${Math.round(cnt/max*100)}%"></div></div>
        <div class="num">${cnt}</div></div>`;
    }
  }
  if (status?.all_runs) {
    const tbody = document.getElementById('dash-runs-body');
    tbody.innerHTML = status.all_runs.slice(0, 5).map(r =>
      `<tr><td>#${r.id}</td><td>${r.mode}</td>
       <td><span class="badge ${r.status==='completed'?'badge-green':r.status==='running'?'badge-yellow':'badge-red'}">${r.status}</span></td>
       <td>${((r.best_val_acc||0)*100).toFixed(1)}%</td></tr>`
    ).join('');
  }
}

// ── Dataset ───────────────────────────────────────────────────────────────────
async function loadDatasetStats() {
  const stats = await fetch(API + '/api/dataset/stats').then(r => r.json()).catch(() => null);
  if (!stats) return;
  const div = document.getElementById('ds-split-table');
  const classes = stats.classes || [];
  let html = `<table class="tbl"><thead><tr><th>Split</th><th>Total</th>${classes.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>`;
  for (const [split, info] of Object.entries(stats.splits||{})) {
    html += `<tr><td>${split}</td><td><strong>${info.total}</strong></td>${classes.map(c=>`<td>${info.per_class?.[c]||0}</td>`).join('')}</tr>`;
  }
  div.innerHTML = html + '</tbody></table>';
}

async function loadThumbnails() {
  const split = document.getElementById('thumb-split')?.value || 'train';
  const data = await fetch(`${API}/api/dataset/thumbnails?split=${split}&max_per_class=6`).then(r=>r.json()).catch(()=>null);
  if (!data) return;
  const div = document.getElementById('thumb-container');
  div.innerHTML = '';
  for (const [cls, imgs] of Object.entries(data)) {
    if (!imgs.length) continue;
    div.innerHTML += `<div style="margin-bottom:1rem"><div style="font-size:.78rem;font-weight:600;color:var(--muted);margin-bottom:.4rem">${cls}</div>
      <div class="thumb-grid">${imgs.map(src=>`<img src="${src}" alt="${cls}"/>`).join('')}</div></div>`;
  }
}

async function generateSynthetic() {
  const r = await fetch(API+'/api/dataset/generate-synthetic', {method:'POST'}).then(r=>r.json()).catch(()=>null);
  if (r) { alert('✅ Synthetic dataset generated: ' + r.stats?.total + ' images'); loadDatasetStats(); loadThumbnails(); }
}

// ── Training ──────────────────────────────────────────────────────────────────
function initCharts() {
  const commonOpts = { responsive:true, animation:false, plugins:{legend:{labels:{color:'#e2f0e2',font:{size:11}}}}, scales:{x:{ticks:{color:'#587058'},grid:{color:'rgba(26,45,26,.5)'}},y:{ticks:{color:'#587058'},grid:{color:'rgba(26,45,26,.5)'}}} };
  if (lossChart) lossChart.destroy();
  if (accChart) accChart.destroy();
  lossChart = new Chart(document.getElementById('chart-loss'), {
    type:'line', data:{labels:[],datasets:[
      {label:'Train Loss',data:[],borderColor:'#f87171',tension:.3,pointRadius:2},
      {label:'Val Loss',data:[],borderColor:'#22c55e',tension:.3,pointRadius:2}]},
    options:{...commonOpts}});
  accChart = new Chart(document.getElementById('chart-acc'), {
    type:'line', data:{labels:[],datasets:[
      {label:'Train Acc',data:[],borderColor:'#60a5fa',tension:.3,pointRadius:2},
      {label:'Val Acc',data:[],borderColor:'#4ade80',tension:.3,pointRadius:2}]},
    options:{...commonOpts}});
}

function addChartPoint(epoch, tl, vl, ta, va) {
  if (!lossChart || !accChart) return;
  lossChart.data.labels.push(epoch);
  lossChart.data.datasets[0].data.push(tl);
  lossChart.data.datasets[1].data.push(vl);
  lossChart.update('none');
  accChart.data.labels.push(epoch);
  accChart.data.datasets[0].data.push(ta);
  accChart.data.datasets[1].data.push(va);
  accChart.update('none');
}

function addEpochRow(msg) {
  const tbody = document.getElementById('epoch-tbl-body');
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${msg.epoch}</td><td><span class="badge badge-green">${msg.phase}</span></td>
    <td>${msg.train_loss}</td><td>${msg.val_loss}</td>
    <td>${msg.train_acc}%</td><td>${msg.val_acc}%</td>
    <td>${msg.learning_rate}</td><td>${msg.co2_kg}</td>`;
  tbody.insertBefore(tr, tbody.firstChild);
}

async function startTraining() {
  const cfg = {
    epochs: +document.getElementById('cfg-epochs').value,
    learning_rate: +document.getElementById('cfg-lr').value,
    batch_size: +document.getElementById('cfg-batch').value,
    fine_tune: document.getElementById('cfg-finetune').value === 'true',
    fine_tune_epochs: +document.getElementById('cfg-ft-epochs').value,
  };
  document.getElementById('btn-start-train').disabled = true;
  document.getElementById('epoch-tbl-body').innerHTML = '';
  const r = await fetch(API+'/api/train/centralized',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)}).then(r=>r.json()).catch(e=>({error:e.message}));
  if (r.error) { alert('Error: '+r.error); document.getElementById('btn-start-train').disabled=false; }
}

async function stopTraining() {
  await fetch(API+'/api/train/stop',{method:'POST'});
}

async function loadTrainHistory() {
  const status = await fetch(API+'/api/train/status').then(r=>r.json()).catch(()=>null);
  if (!status?.history?.length) return;
  initCharts();
  document.getElementById('train-progress-card').style.display='';
  document.getElementById('epoch-tbl-body').innerHTML='';
  for (const m of status.history) { addChartPoint(m.epoch,m.train_loss,m.val_loss,m.train_acc,m.val_acc); addEpochRow(m); }
}

// ── Learning Viewer ───────────────────────────────────────────────────────────
let viewerFile = null;
function viewerPreview(e) {
  viewerFile = e.target.files[0];
  if (!viewerFile) return;
  const url = URL.createObjectURL(viewerFile);
  const orig = document.getElementById('viewer-orig');
  orig.src = url; orig.style.display='block';
  document.getElementById('viewer-result').innerHTML='';
  document.getElementById('heatmap-img').style.display='none';
  document.getElementById('heatmap-placeholder').style.display='flex';
  // Auto-run predict
  runViewerPredict();
}

async function runViewerPredict() {
  if (!viewerFile) return;
  const spin = document.getElementById('viewer-spin');
  spin.style.display='block';
  const form = new FormData();
  form.append('file', viewerFile);
  try {
    const data = await fetch(API+'/predict',{method:'POST',body:form}).then(r=>r.json());
    const pct = Math.round(data.confidence*100);
    let html = `<div style="font-size:1.2rem;font-weight:700;color:var(--green);margin-bottom:.5rem">${data.predicted_class}</div>
      <div class="pbar-bg" style="margin-bottom:.75rem"><div class="pbar-fill" style="width:${pct}%"></div></div>`;
    if (data.category_scores) {
      const sorted = Object.entries(data.category_scores).sort((a,b)=>b[1]-a[1]);
      html += sorted.map(([c,s])=>`<div class="cls-bar">
        <div class="name" style="width:160px">${c}</div>
        <div class="bar-bg"><div class="bar-fill" style="width:${Math.round(s*100)}%"></div></div>
        <div class="num">${Math.round(s*100)}%</div></div>`).join('');
    }
    html += `<div style="margin-top:.5rem;font-size:.75rem;color:var(--muted)">${data.description||''}</div>`;
    html += `<div style="margin-top:.3rem;font-size:.7rem;color:var(--muted)">Model: ${data.model_version||'imagenet'} · ${data.inference_time_ms}ms</div>`;
    document.getElementById('viewer-result').innerHTML=html;
    if (data.gradcam_b64) {
      const img = document.getElementById('heatmap-img');
      img.src = data.gradcam_b64; img.style.display='block';
      document.getElementById('heatmap-placeholder').style.display='none';
    }
  } catch(e) {
    document.getElementById('viewer-result').innerHTML='<span style="color:var(--red)">API error: '+e.message+'</span>';
  }
  spin.style.display='none';
}

// ── FL ────────────────────────────────────────────────────────────────────────
async function startFL() {
  const rounds = document.getElementById('fl-rounds').value;
  const nodes = document.getElementById('fl-nodes').value;
  const r = await fetch(`${API}/api/fl/start?rounds=${rounds}&num_nodes=${nodes}`,{method:'POST'}).then(r=>r.json()).catch(e=>({error:e.message}));
  if (r.error) return alert('Error: '+r.error);
  alert(`FL started! Server PID: ${r.server_pid}. Node PIDs: ${r.node_pids?.join(', ')}`);
  for (let i=0;i<3;i++){const el=document.getElementById('fl-n'+i+'-status');if(el){el.textContent='Waiting';el.className='badge badge-yellow';}}
  setTimeout(loadFLStatus, 5000);
}

async function loadFLStatus() {
  const data = await fetch(API+'/api/fl/status').then(r=>r.json()).catch(()=>null);
  if (!data) return;
  data.nodes?.forEach((n,i)=>{
    const el=document.getElementById('fl-n'+i+'-status');
    if(el){el.textContent=n.running?'Running':'Done';el.className='badge '+(n.running?'badge-green':'badge-yellow');}
  });
  const tbody=document.getElementById('fl-rounds-body');
  if(data.rounds?.length){
    tbody.innerHTML=data.rounds.map(r=>`<tr><td>${r.round_num}</td><td>${r.node_id}</td><td>${r.local_loss?.toFixed(4)||'—'}</td><td>${((r.local_acc||0)*100).toFixed(1)}%</td><td>${r.co2_kg?.toFixed(6)||'—'}</td><td>${r.bandwidth_mb?.toFixed(3)||'—'}</td></tr>`).join('');
  }
}

function addFLRoundRow(msg) {
  const tbody = document.getElementById('fl-rounds-body');
  tbody.innerHTML = `<tr><td>${msg.round}</td><td>server</td><td>—</td><td>—</td><td>${msg.server_co2_kg}</td><td>—</td></tr>` + tbody.innerHTML;
}

// ── Models ────────────────────────────────────────────────────────────────────
async function loadModels() {
  const data = await fetch(API+'/api/models').then(r=>r.json()).catch(()=>[]);
  const tbody = document.getElementById('models-body');
  if (!data.length) { tbody.innerHTML='<tr><td colspan="7" style="color:var(--muted);text-align:center">No trained models yet. Train one first!</td></tr>'; return; }
  tbody.innerHTML = data.map(m => `<tr>
    <td><strong>${m.version}</strong></td>
    <td>${((m.train_acc||0)*100).toFixed(1)}%</td>
    <td>${((m.val_acc||0)*100).toFixed(1)}%</td>
    <td>${(m.co2_kg||0).toFixed(6)}</td>
    <td style="font-size:.72rem;color:var(--muted)">${(m.created_at||'').slice(0,16)}</td>
    <td>${m.is_active?'<span class="badge badge-green">Active</span>':'<span class="badge badge-yellow">Inactive</span>'}</td>
    <td style="display:flex;gap:.4rem">
      <button class="btn btn-outline" style="padding:.3rem .6rem;font-size:.72rem" onclick="activateModel('${m.version}')">✅ Activate</button>
      <a href="${API}/api/models/${m.version}/download" class="btn btn-outline" style="padding:.3rem .6rem;font-size:.72rem">⬇ Download</a>
    </td></tr>`).join('');
}

async function activateModel(version) {
  await fetch(`${API}/api/models/${version}/activate`,{method:'POST'});
  loadModels();
}

// ── Predict ───────────────────────────────────────────────────────────────────
function loadPredictFile(e) {
  predFile = e.target.files[0];
  if (!predFile) return;
  document.getElementById('pred-drop-zone').style.display='none';
  document.getElementById('pred-preview-row').style.display='flex';
  document.getElementById('pred-img').src = URL.createObjectURL(predFile);
  document.getElementById('pred-result').style.display='none';
  document.getElementById('pred-error').style.display='none';
}
function resetPredict() {
  predFile=null;
  document.getElementById('pred-drop-zone').style.display='';
  document.getElementById('pred-preview-row').style.display='none';
}
async function runPredict() {
  if (!predFile) return;
  const spin=document.getElementById('pred-spin'), btn=document.getElementById('btn-predict');
  spin.style.display='block'; btn.disabled=true;
  document.getElementById('pred-result').style.display='none';
  document.getElementById('pred-error').style.display='none';
  try {
    const form=new FormData(); form.append('file',predFile);
    const data = await fetch(API+'/predict',{method:'POST',body:form}).then(r=>r.json());
    const pct=Math.round(data.confidence*100);
    document.getElementById('pred-class').textContent=data.predicted_class;
    setTimeout(()=>{ document.getElementById('pred-conf-bar').style.width=pct+'%'; },50);
    let scoresHtml='';
    if (data.category_scores) {
      const sorted=Object.entries(data.category_scores).sort((a,b)=>b[1]-a[1]);
      scoresHtml=sorted.map(([c,s])=>`<div class="cls-bar">
        <div class="name" style="width:160px;font-size:.75rem">${c}</div>
        <div class="bar-bg"><div class="bar-fill" style="width:${Math.round(s*100)}%"></div></div>
        <div class="num" style="font-size:.72rem">${Math.round(s*100)}%</div></div>`).join('');
    }
    document.getElementById('pred-scores').innerHTML=scoresHtml;
    document.getElementById('pred-desc').textContent=data.description||'';
    document.getElementById('pred-meta').textContent=`Model: ${data.model_version||'imagenet'} · Inference: ${data.inference_time_ms}ms · Confidence: ${(data.confidence*100).toFixed(2)}%`;
    document.getElementById('pred-result').style.display='block';
  } catch(e) {
    document.getElementById('pred-error').textContent='⚠ '+e.message;
    document.getElementById('pred-error').style.display='block';
  }
  spin.style.display='none'; btn.disabled=false;
}

// ── Compare & Report ──────────────────────────────────────────────────────────
async function loadCompare() {
  const [cmp, runs] = await Promise.all([
    fetch(API+'/api/compare').then(r=>r.json()).catch(()=>null),
    fetch(API+'/api/train/status').then(r=>r.json()).catch(()=>null),
  ]);
  if (cmp?.centralized) {
    document.getElementById('cmp-central').innerHTML = renderCompareCard(cmp.centralized);
  }
  if (cmp?.federated) {
    document.getElementById('cmp-fed').innerHTML = renderCompareCard(cmp.federated);
  }
  if (runs?.all_runs) {
    document.getElementById('all-runs-body').innerHTML = runs.all_runs.map(r=>
      `<tr><td>#${r.id}</td><td>${r.mode}</td>
       <td><span class="badge ${r.status==='completed'?'badge-green':r.status==='running'?'badge-yellow':'badge-red'}">${r.status}</span></td>
       <td>${((r.best_val_acc||0)*100).toFixed(1)}%</td><td>${r.total_epochs}</td>
       <td style="font-size:.72rem;color:var(--muted)">${(r.started_at||'').slice(0,16)}</td></tr>`
    ).join('');
    if (runs.all_runs.length > 0) {
      document.getElementById('report-run-id').value = runs.all_runs[0].id;
    }
  }
}

function renderCompareCard(d) {
  return `<div class="stat" style="margin-bottom:.5rem">
    <div class="val">${d.best_val_acc}%</div><div class="lbl">Best Val Accuracy</div></div>
    <div style="font-size:.78rem;color:var(--muted)">${d.run_count} run(s) recorded</div>`;
}

function dlReport(type) {
  const id = document.getElementById('report-run-id').value;
  window.open(`${API}/api/report/${type}/${id}`, '_blank');
}

// ── Quick Guide ───────────────────────────────────────────────────────────────
function openGuide() {
  document.getElementById('dont-show-again').checked = localStorage.getItem('hideGreenScanGuide') === 'true';
  document.getElementById('guide-modal').classList.add('active');
}

function closeGuide() {
  const hide = document.getElementById('dont-show-again').checked;
  if (hide) {
    localStorage.setItem('hideGreenScanGuide', 'true');
  } else {
    localStorage.removeItem('hideGreenScanGuide');
  }
  document.getElementById('guide-modal').classList.remove('active');
}

// ── Live Camera Predict ───────────────────────────────────────────────────────
let cameraStream = null;

async function startCamera() {
  const video = document.getElementById('camera-stream');
  const btnStart = document.getElementById('btn-start-camera');
  const btnCapture = document.getElementById('btn-capture-predict');
  const status = document.getElementById('cam-status');

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    video.srcObject = cameraStream;
    btnStart.textContent = "📷 Stop Camera";
    btnStart.classList.remove('btn-outline');
    btnStart.classList.add('btn-red');
    btnStart.onclick = stopCamera;
    btnCapture.disabled = false;
    status.textContent = "Camera active. Ready to capture.";
  } catch (err) {
    status.textContent = "Error: " + err.message;
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach(track => track.stop());
    cameraStream = null;
  }
  const video = document.getElementById('camera-stream');
  video.srcObject = null;
  
  const btnStart = document.getElementById('btn-start-camera');
  const btnCapture = document.getElementById('btn-capture-predict');
  const status = document.getElementById('cam-status');
  
  btnStart.textContent = "📷 Start Camera";
  btnStart.classList.remove('btn-red');
  btnStart.classList.add('btn-outline');
  btnStart.onclick = startCamera;
  btnCapture.disabled = true;
  status.textContent = "Camera inactive.";
}

async function captureAndPredict() {
  const video = document.getElementById('camera-stream');
  const canvas = document.getElementById('camera-canvas');
  const status = document.getElementById('cam-status');
  const spin = document.getElementById('cam-spin');
  const container = document.getElementById('cam-predict-container');
  
  if (!cameraStream) return;
  
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
  container.style.display = 'none';
  spin.style.display = 'block';
  status.textContent = "Analyzing image...";
  
  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append('file', blob, 'camera_capture.jpg');
    
    try {
      const res = await fetch(`${API}/predict`, { method: 'POST', body: formData });
      const data = await res.json();
      
      if (!res.ok) throw new Error(data.detail || "Prediction failed");
      
      document.getElementById('cam-snapshot').src = canvas.toDataURL('image/jpeg');
      document.getElementById('cam-class').textContent = data.predicted_class;
      document.getElementById('cam-conf-bar').style.width = (data.confidence * 100) + '%';
      
      let html = '';
      if (data.top5) {
        data.top5.forEach(t => {
          const w = (t.prob * 100).toFixed(1);
          html += `<div class="cls-bar"><div class="name">${t.class}</div><div class="bar-bg"><div class="bar-fill" style="width:${w}%"></div></div><div class="num">${w}%</div></div>`;
        });
      }
      document.getElementById('cam-scores').innerHTML = html;
      document.getElementById('cam-desc').textContent = data.description || '';
      document.getElementById('cam-meta').textContent = `Inference: ${data.inference_time_ms}ms | Model: ${data.model_version}`;
      
      spin.style.display = 'none';
      container.style.display = 'block';
      status.textContent = "Prediction complete.";
    } catch (err) {
      spin.style.display = 'none';
      status.textContent = "Error: " + err.message;
    }
  }, 'image/jpeg', 0.9);
}

// ── Boot ──────────────────────────────────────────────────────────────────────
connectWS();
loadDashboard();

// Show guide on first startup
if (localStorage.getItem('hideGreenScanGuide') !== 'true') {
  setTimeout(openGuide, 600);
}

