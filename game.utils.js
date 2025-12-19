// ========== 14) 工具 ==========
function clearSelections(){
  ui.selectedTokenColors.clear();
  ui.selectedMarketCardId = null;
  ui.selectedReservedCard = null;
}

function toast(msg, { type = "info" } = {}){
  console.log("[toast]", msg);
  if (type === "error"){
    return showStatusMessage(msg, { type });
  }
}

function showStatusMessage(msg, { type = "info" } = {}){
  ui.errorMessage = msg;
  if (typeof renderErrorBanner === "function"){
    renderErrorBanner();
  }
}

function shuffle(arr){
  const a = [...arr];
  for (let i=a.length-1;i>0;i--){
    const j = Math.floor(Math.random()*(i+1));
    [a[i],a[j]]=[a[j],a[i]];
  }
  return a;
}
function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"
  }[c]));
}

let timerIntervalId = null;

function formatDuration(ms = 0){
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  const parts = [
    hours > 0 ? String(hours).padStart(2, "0") : null,
    String(minutes).padStart(2, "0"),
    String(seconds).padStart(2, "0"),
  ].filter(Boolean);
  return parts.join(":");
}

function computeElapsedMs(){
  if (!state.gameStartAt) return 0;
  const end = state.gameEndAt || Date.now();
  return Math.max(0, end - state.gameStartAt);
}

function renderGameTimer(){
  if (!el.gameTimer) return;
  if (!state.gameStartAt){
    stopGameTimer();
    el.gameTimer.textContent = "--:--";
    return;
  }
  el.gameTimer.textContent = formatDuration(computeElapsedMs());
}

function stopGameTimer(){
  if (timerIntervalId){
    clearInterval(timerIntervalId);
    timerIntervalId = null;
  }
}

function startGameTimer(){
  stopGameTimer();
  state.gameStartAt = Date.now();
  state.gameEndAt = null;
  renderGameTimer();
  ensureTimerInterval();
}

function resumeGameTimer(){
  if (!state.gameStartAt) maybeEnsureGameStartTime();
  renderGameTimer();
  ensureTimerInterval();
}

function maybeEnsureGameStartTime(){
  if (state.gameStartAt) return;
  const createdMs = state.createdAt ? new Date(state.createdAt).getTime() : NaN;
  const fallback = Number.isFinite(createdMs) ? createdMs : Date.now();
  state.gameStartAt = fallback;
  state.gameEndAt = null;
}

function ensureTimerInterval(){
  if (state.victoryResolved || state.gameEndAt){
    stopGameTimer();
    return;
  }
  if (state.gameStartAt && !timerIntervalId){
    timerIntervalId = setInterval(renderGameTimer, 1000);
  }
}
