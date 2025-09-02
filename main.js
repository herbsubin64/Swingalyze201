import { PoseLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d", { alpha: true });

const fileInput = $("file");
const demoBtn = $("demo");
const playBtn = $("playpause");
const resetBtn = $("reset");
const overlayBtn = $("toggleOverlay");
const notesEl = $("notes");
const fpsEl = $("fps");
const spineDegEl = $("spineDeg");
const headMoveEl = $("headMove");
const swayEl = $("sway");
const kneesEl = $("knees");
const xfactorEl = $("xfactor");
const mirrorEl = $("mirror");
const slowEl = $("slow");

let pose = null;
let overlayOn = true;
let running = false;
let lastT = 0, fps = 0;
let drawRect = { x: 0, y: 0, w: 0, h: 0, scaleX: 1, scaleY: 1 };
let baseline = null; // for head/pelvis drift
let phase = "address"; // naive swing phase tracker

// ——— Initialize pose ———
(async function init() {
  try {
    status("Loading models…");
    const fileset = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    pose = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
      outputSegmentationMasks: false,
    });
    status("Ready. Load a video.");
  } catch (e) {
    status("Failed to load models — check network/CORS.");
    console.error(e);
  }
})();

// ——— Video wiring ———
fileInput.addEventListener("change", (e) => {
  const f = e.target.files?.[0];
  if (!f) return;
  loadVideo(URL.createObjectURL(f));
});
demoBtn.addEventListener("click", () => {
  // Simple public clip just to prove playback/fit; not golf.
  loadVideo("https://interactive-examples.mdn.mozilla.net/media/cc0-videos/flower.mp4");
});
playBtn.addEventListener("click", () => {
  if (!video.src) return;
  if (video.paused) video.play(); else video.pause();
});
resetBtn.addEventListener("click", resetSession);
overlayBtn.addEventListener("click", () => { overlayOn = !overlayOn; });

["play", "pause"].forEach(ev => video.addEventListener(ev, () => {
  playBtn.textContent = video.paused ? "Play" : "Pause";
}));

video.addEventListener("loadeddata", () => {
  computeDrawRect();
  canvas.width = Math.round(drawRect.w);
  canvas.height = Math.round(drawRect.h);
  running = true;
  baseline = null;
  phase = "address";
  video.play().catch(()=>{});
  requestAnimationFrame(tick);
});

window.addEventListener("resize", () => {
  computeDrawRect();
  canvas.width = Math.round(drawRect.w);
  canvas.height = Math.round(drawRect.h);
});

function loadVideo(src) {
  resetSession();
  video.src = src;
  video.crossOrigin = "anonymous";
  video.muted = true;
  video.playsInline = true;
}

// Keep stage aspect-ratio with object-fit: contain mapping
function computeDrawRect() {
  const stage = video.parentElement; // .stage
  const W = stage.clientWidth;
  const H = stage.clientHeight;
  const vw = video.videoWidth || 16;
  const vh = video.videoHeight || 9;
  const aspectVideo = vw / vh;
  const aspectStage = W / H;

  let w, h;
  if (aspectVideo > aspectStage) {
    w = W; h = W / aspectVideo;
  } else {
    h = H; w = H * aspectVideo;
  }
  const x = (W - w) / 2;
  const y = (H - h) / 2;
  drawRect = { x, y, w, h, scaleX: w / vw, scaleY: h / vh };
  // Position DOM video & canvas to center; canvas sized separately.
  video.style.width = `${w}px`; video.style.height = `${h}px`;
  video.style.left = `${x + w/2}px`; video.style.top = `${y + h/2}px`;
  canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
  canvas.style.left = `${x + w/2}px`; canvas.style.top = `${y + h/2}px`;
}

// ——— Main loop ———
function tick(ts) {
  if (!running || !pose) return;
  requestAnimationFrame(tick);

  const dt = ts - lastT;
  if (dt >= 250) { // update status even if throttled
    fps = 1000 / Math.max(1, dt);
    fpsEl.textContent = fps.toFixed(1);
    lastT = ts;
  }

  const t = video.currentTime;
  if (video.paused || video.ended) return;
  if (slowEl.checked) video.playbackRate = 0.5; else video.playbackRate = 1.0;

  // Pose estimation
  const res = pose.detectForVideo(video, ts);
  const lm = res?.landmarks?.[0];
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!lm) { status("No pose detected"); return; }
  status("Analyzing…");

  // Map landmarks to canvas space (contain fit + optional mirror)
  const pts = mapLandmarks(lm);
  if (overlayOn) drawSkeleton(pts);

  // Measurements (robust to missing pts)
  const m = measurements(pts);
  spineDegEl.textContent = isNum(m.spineAngle) ? m.spineAngle.toFixed(1) : "–";
  headMoveEl.textContent = isNum(m.headDriftPx) ? m.headDriftPx.toFixed(0) : "–";
  swayEl.textContent = isNum(m.pelvisDriftPx) ? m.pelvisDriftPx.toFixed(0) : "–";
  kneesEl.textContent = isNum(m.kneeFlex) ? m.kneeFlex.toFixed(1) : "–";
  xfactorEl.textContent = isNum(m.xFactor) ? m.xFactor.toFixed(1) : "–";

  // Phase tracker & notes
  const tips = analyzePhases(m);
  renderNotes(tips);
}

function mapLandmarks(lm) {
  // Indices per MediaPipe Pose
  // https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
  const vw = video.videoWidth, vh = video.videoHeight;
  const sx = drawRect.scaleX, sy = drawRect.scaleY;
  const mirror = mirrorEl.checked ? -1 : 1;
  const offx = mirrorEl.checked ? drawRect.w : 0;

  const pts = lm.map(p => ({
    x: Math.round(p.x * vw * sx * mirror + offx),
    y: Math.round(p.y * vh * sy),
    z: p.z
  }));
  return pts;
}

function drawSkeleton(pts) {
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#6fb1fc";
  ctx.globalAlpha = 0.9;
  const C = (i) => pts[i];

  // helper connect
  const conn = (a,b) => {
    if (!C(a) || !C(b)) return;
    ctx.beginPath();
    ctx.moveTo(C(a).x, C(a).y);
    ctx.lineTo(C(b).x, C(b).y);
    ctx.stroke();
  };

  // Torso
  conn(11,12); conn(11,23); conn(12,24); conn(23,24);
  // Left arm
  conn(11,13); conn(13,15);
  // Right arm
  conn(12,14); conn(14,16);
  // Legs
  conn(23,25); conn(25,27);
  conn(24,26); conn(26,28);
  // Shoulders to neck proxy: mid(11,12) to nose(0)
  const sMid = mid(pts[11], pts[12]);
  if (sMid && pts[0]) { ctx.beginPath(); ctx.moveTo(sMid.x, sMid.y); ctx.lineTo(pts[0].x, pts[0].y); ctx.stroke(); }
}

function isNum(v){ return Number.isFinite(v); }
function mid(a,b){ if(!a||!b) return null; return { x:(a.x+b.x)/2, y:(a.y+b.y)/2 }; }
function deg(rad){ return rad*180/Math.PI; }

function measurements(pts){
  const m = {};
  const L = (i)=>pts[i];
  const hipMid = mid(L(23), L(24));
  const shMid = mid(L(11), L(12));
  const earL = L(7), earR = L(8);
  const kneeL = L(25), kneeR = L(26);
  const ankleL = L(27), ankleR = L(28);
  const nose = L(0);

  // spine angle (line hipMid->shMid against vertical)
  if (hipMid && shMid) {
    const dx = shMid.x - hipMid.x;
    const dy = shMid.y - hipMid.y;
    m.spineAngle = deg(Math.atan2(dx, dy)); // positive = tilt away
  }

  // establish baseline at start of play
  if (!baseline && hipMid && nose) {
    baseline = { headX: nose.x, pelvisX: hipMid.x, t: performance.now() };
  }
  if (baseline && nose) m.headDriftPx = nose.x - baseline.headX;
  if (baseline && hipMid) m.pelvisDriftPx = hipMid.x - baseline.pelvisX;

  // knee flex: average thigh vs shank angle
  if (kneeL && ankleL && hipMid && kneeR && ankleR) {
    // Left knee angle
    const thighL = Math.atan2(kneeL.y - hipMid.y, kneeL.x - hipMid.x);
    const shankL = Math.atan2(ankleL.y - kneeL.y, ankleL.x - kneeL.x);
    const kneeAngleL = Math.abs(deg(shankL - thighL));
    // Right knee angle
    const thighR = Math.atan2(kneeR.y - hipMid.y, kneeR.x - hipMid.x);
    const shankR = Math.atan2(ankleR.y - kneeR.y, ankleR.x - kneeR.x);
    const kneeAngleR = Math.abs(deg(shankR - thighR));
    m.kneeFlex = 180 - ((kneeAngleL + kneeAngleR) / 2);
  }

  // X‑factor: shoulders vs hips rotation proxy using horizontal vector
  if (L(11) && L(12) && L(23) && L(24)) {
    const sh = Math.atan2(L(12).y - L(11).y, L(12).x - L(11).x);
    const hp = Math.atan2(L(24).y - L(23).y, L(24).x - L(23).x);
    m.xFactor = Math.abs(deg(sh - hp));
  }

  return m;
}

function analyzePhases(m){
  const tips = [];
  // Simple thresholds tuned for pixel space (canvas)
  if (phase === "address") {
    tips.push(msg("At address: hold steady head & pelvis. Target < 10px drift."));
    if (Math.abs(m.headDriftPx || 0) > 10) tips.push(bad("Head swaying at address."));
    if (Math.abs(m.pelvisDriftPx || 0) > 10) tips.push(warn("Pelvis shifting; quiet lower body."));
    // Proceed once playback proceeds a bit
    phase = "backswing";
  }
  else if (phase === "backswing") {
    if (isNum(m.xFactor) && m.xFactor > 30) tips.push(good("Good shoulder‑hip separation."));
    if (isNum(m.kneeFlex) && m.kneeFlex < 10) tips.push(warn("Maintain some knee flex in backswing."));
    tips.push(msg("Top of backswing: check stable base, growing X‑factor."));
    phase = "downswing";
  }
  else if (phase === "downswing") {
    if (isNum(m.pelvisDriftPx) && Math.abs(m.pelvisDriftPx) > 20) tips.push(good("Initiate with hips — weight shift detected."));
    tips.push(msg("Create sequence: hips → torso → arms."));
    phase = "impact";
  }
  else if (phase === "impact") {
    if (isNum(m.spineAngle) && Math.abs(m.spineAngle) < 5) tips.push(good("Neutral spine at impact."));
    tips.push(msg("Hands ahead of clubhead (forward shaft lean)."));
    phase = "follow";
  }
  else if (phase === "follow") {
    tips.push(msg("Balanced finish; chest to target."));
  }
  return tips;
}

function renderNotes(items){
  // de‑dupe and limit
  const key = (o)=>o.text;
  const seen = new Set();
  const lim = 6;
  const list = [];
  for (const it of items) {
    const k = key(it);
    if (!seen.has(k)) { seen.add(k); list.push(it); if (list.length>=lim) break; }
  }
  notesEl.innerHTML = list.map(it => `<li class="note ${it.kind}">${escapeHtml(it.text)}</li>`).join("");
}

function msg(t){ return { kind:"", text:t }; }
function good(t){ return { kind:"good", text:t }; }
function warn(t){ return { kind:"warn", text:t }; }
function bad(t){ return { kind:"bad", text:t }; }

function escapeHtml(s){ return s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function resetSession(){
  running = false;
  ctx.clearRect(0,0,canvas.width, canvas.height);
  notesEl.innerHTML = "";
  fpsEl.textContent = "–";
  spineDegEl.textContent = headMoveEl.textContent = swayEl.textContent = kneesEl.textContent = xfactorEl.textContent = "–";
  status("Ready.");
}

function status(s){ statusEl.textContent = s; }