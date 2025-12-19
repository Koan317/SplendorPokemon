// ========== 3) 游戏状态（存档核心） ==========
/**
 * 存档包含：
 * - 每个玩家：hand(手牌区/桌面阵列)、reserved(保留区)、tokens(token区)、name、isStarter
 */
let state = makeEmptyState();

let ui = {
  selectedTokenColors: new Set(), // for take actions
  selectedMarketCardId: null,     // for reserve/buy
  selectedReservedCard: null,     // {playerIndex, cardId}
  handPreviewPlayerIndex: null,
  errorMessage: "",
  tokenReturn: null,              // { playerIndex, required, selected: number[6] }
  hasActiveGame: false,
  gameTimerIntervalId: null,
};

const wait = (ms = 0) => new Promise(resolve => setTimeout(resolve, ms));
const ensurePromise = (value) => (value && typeof value.then === "function") ? value : Promise.resolve(value);

let cardLibraryData = null;

function makeEmptyState(){
  return {
    version: 1,
    createdAt: new Date().toISOString(),
    finishedAt: null,
    // 公共区（不要求存档也可以存，建议存：方便完全复现）
    tokenPool: [7,7,7,7,7,5], // 默认 4人
    market: {
      slotsByLevel: { 1: [], 2: [], 3: [], 4: [], 5: [] },
    },
    decks: {
      lv1: [],
      lv2: [],
      lv3: [],
      rare: [],
      legend: [],
    },

    players: [],

    turn: 1,
    currentPlayerIndex: 0,
    // 规则约束（每回合一次进化）
    perTurn: {
      evolved: false,
      primaryAction: null,
    },
    endTriggered: false,
    endTriggerTurn: null,
    victoryResolved: false,
  };
}

function ensurePerTurnDefaults(){
  if (!state.perTurn) state.perTurn = { evolved: false, primaryAction: null };
  if (state.perTurn.evolved === undefined) state.perTurn.evolved = false;
  if (state.perTurn.primaryAction === undefined) state.perTurn.primaryAction = null;
}

function stopGameTimer(){
  if (ui.gameTimerIntervalId){
    clearInterval(ui.gameTimerIntervalId);
    ui.gameTimerIntervalId = null;
  }
}

function formatDuration(totalSeconds){
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return [
    hours.toString().padStart(2, "0"),
    minutes.toString().padStart(2, "0"),
    seconds.toString().padStart(2, "0"),
  ].join(":");
}

function getElapsedSeconds(){
  if (!ui.hasActiveGame) return null;
  if (!state?.createdAt) return null;
  const startTs = Date.parse(state.createdAt);
  if (!Number.isFinite(startTs)) return null;
  const endTs = state.finishedAt ? Date.parse(state.finishedAt) : Date.now();
  if (!Number.isFinite(endTs)) return null;
  return Math.max(0, Math.floor((endTs - startTs) / 1000));
}

function updateGameTimerDisplay(){
  if (!el.gameTimer) return;
  const elapsedSeconds = getElapsedSeconds();
  if (elapsedSeconds === null){
    el.gameTimer.textContent = "--:--";
    return;
  }
  el.gameTimer.textContent = formatDuration(elapsedSeconds);
}

function startGameTimer(){
  ui.hasActiveGame = true;
  state.finishedAt = null;
  stopGameTimer();
  updateGameTimerDisplay();
  ui.gameTimerIntervalId = setInterval(updateGameTimerDisplay, 1000);
}

function finishGameTimer(){
  if (!ui.hasActiveGame) return;
  if (!state.finishedAt){
    state.finishedAt = new Date().toISOString();
  }
  stopGameTimer();
  updateGameTimerDisplay();
}

function syncGameTimerFromState(){
  ui.hasActiveGame = true;
  if (state.victoryResolved && !state.finishedAt){
    state.finishedAt = new Date().toISOString();
  }
  stopGameTimer();
  updateGameTimerDisplay();
  if (!state.finishedAt){
    ui.gameTimerIntervalId = setInterval(updateGameTimerDisplay, 1000);
  }
}

function getPlayerAiLevel(player, index){
  if (!player) return DISABLED_AI_LEVEL;
  if (typeof player.aiLevel === "number") return player.aiLevel;
  return index === 0 ? DISABLED_AI_LEVEL : DEFAULT_AI_LEVEL;
}

function ensurePlayerHasAiLevel(player, index){
  if (!player) return;
  player.aiLevel = getPlayerAiLevel(player, index);
}

function isAIControlledPlayer(index){
  return getPlayerAiLevel(state.players[index], index) >= 0 && index > 0;
}

function getPrimaryActionLabel(key){
  return PRIMARY_ACTION_LABELS[key] || "主要行动";
}

function hasTakenPrimaryAction(){
  ensurePerTurnDefaults();
  return !!state.perTurn.primaryAction;
}

function blockIfPrimaryActionLocked(){
  if (!hasTakenPrimaryAction()) return false;
  toast(`本回合已执行【${getPrimaryActionLabel(state.perTurn.primaryAction)}】，请结束回合后再进行其他主要行动`, { type: "error" });
  return true;
}

function markPrimaryAction(actionKey){
  ensurePerTurnDefaults();
  state.perTurn.primaryAction = actionKey;
}
