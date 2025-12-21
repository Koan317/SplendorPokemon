// ========== 5) 新游戏初始化 ==========
async function newGame(playerCount){
  const loadedLib = await loadCardLibrary();
  const lib = prepareLibraryForPlayerCount(loadedLib, playerCount);
  lastLoadError = null;
  state = makeEmptyState();
  ui.errorMessage = "";
  state.createdAt = new Date().toISOString();
  state.sessionEndedAt = null;

  state.tokenPool = makeTokenPoolByPlayerCount(playerCount);

  state.players = [];
  for (let i=0;i<playerCount;i++){
    state.players.push({
      id: `P${i}`,
      name: i === 0 ? "玩家" : `机器人${i}`,
      aiLevel: i === 0 ? DISABLED_AI_LEVEL : DEFAULT_AI_LEVEL,
      isStarter: false,
      hand: [],      // bought/captured cards on table
      reserved: [],  // reserved cards
      tokens: [0,0,0,0,0,0], // counts by color
    });
  }
  state.players[0].isStarter = true;
  state.currentPlayerIndex = 0;
  state.turn = 1;
  state.perTurn = { evolved: false, primaryAction: null };

  state.decks = buildDecksFromLibrary(lib);
  refillMarketFromDecks();

  clearSelections();
  resetSessionTimer();
  renderAll();
}

// token 数量按人数
function makeTokenPoolByPlayerCount(n){
  if (n === 2) return [4,4,4,4,4,5];
  if (n === 3) return [6,6,6,6,6,5];
  if (n === 5) return [8,8,8,8,8,6];
  return [7,7,7,7,7,5];
}

function prepareLibraryForPlayerCount(lib, playerCount){
  if (playerCount !== 5) return lib;
  const colorToCopy = randomIntInclusive(0, 4);
  const cloned = JSON.parse(JSON.stringify(lib || {}));
  return cloneLibraryWithExtraRewardColorCards(cloned, colorToCopy);
}

function randomIntInclusive(min, max){
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function cloneLibraryWithExtraRewardColorCards(lib, ballColor){
  const deckKeys = ["level_1", "level_2", "level_3", "rare", "legend"];
  deckKeys.forEach(key => {
    const cards = lib[key] || [];
    const copies = cards
      .filter(card => card?.reward?.ball_color === ballColor)
      .map((card, idx) => {
        const copy = { ...card };
        const uniqueSuffix = `${ballColor}-${idx}-${Math.random().toString(16).slice(2)}`;
        if (copy.md5){
          copy.md5 = `${copy.md5}-copy-${uniqueSuffix}`;
        } else if (copy.id){
          copy.id = `${copy.id}-copy-${uniqueSuffix}`;
        } else {
          copy.id = `copy-${uniqueSuffix}`;
        }
        return copy;
      });
    lib[key] = [...cards, ...copies];
  });
  return lib;
}

function levelKey(level){
  if (level === 1) return "lv1";
  if (level === 2) return "lv2";
  if (level === 3) return "lv3";
  if (level === 4) return "rare";
  return "legend";
}

function drawFromDeck(level){
  const key = levelKey(level);
  const deck = state.decks[key];
  if (!deck || deck.length === 0) return null;
  return deck.pop();
}

function ensureMarketSlotsByLevel(level){
  const sizes = { 1: 4, 2: 4, 3: 4, 4: 1, 5: 1 };
  const want = sizes[level] || 0;
  const slots = state.market.slotsByLevel[level] || [];
  while (slots.length < want){
    slots.push(null);
  }
  state.market.slotsByLevel[level] = slots;
}

function refillMarketFromDecks(){
  for (const level of [1,2,3,4,5]){
    ensureMarketSlotsByLevel(level);
    const slots = state.market.slotsByLevel[level];
    for (let i=0;i<slots.length;i++){
      if (!slots[i]){
        slots[i] = drawFromDeck(level);
      }
    }
  }
}

function findMarketCard(cardId){
  for (const level of [1,2,3,4,5]){
    const slots = state.market.slotsByLevel[level] || [];
    const idx = slots.findIndex(c => c && c.id === cardId);
    if (idx >= 0){
      return { level, idx, card: slots[idx] };
    }
  }
  return null;
}
