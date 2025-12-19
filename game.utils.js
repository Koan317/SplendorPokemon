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

function formatDuration(ms){
  const totalSeconds = Math.max(0, Math.floor((typeof ms === "number" ? ms : 0) / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  const parts = [];
  if (hours > 0) parts.push(`${hours}小时`);
  parts.push(`${minutes.toString().padStart(2, "0")}分`);
  parts.push(`${seconds.toString().padStart(2, "0")}秒`);
  return parts.join("");
}
