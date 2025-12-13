// 基于现有的 card_image 和 cards.json 生成一个静态展示，方便预览卡牌与 token 布局。
// 需求：
// - 每个人的保留区展示 3 张随机卡牌
// - 每个人的 token 区放置 8 个 token（使用“大师球.png”）
// - 每个人的已获得卡牌展示 10 张随机卡牌
// - 牌桌中央的牌堆与展示位都各展示一张随机卡牌

const TOKEN_IMAGE = "card_image/大师球.png";
const PLAYER_AREAS = ["#player-top", "#player-left", "#player-right", "#player-bottom"];

async function loadCardData() {
  const response = await fetch("cards.json");
  if (!response.ok) {
    throw new Error(`无法读取 cards.json：${response.status}`);
  }
  const data = await response.json();
  return data.filter((card) => card && card.level !== -1 && card.src);
}

function resolveCardImage(card) {
  if (!card || !card.src) return "";
  if (/\.png$/i.test(card.src) || /\.jpg$/i.test(card.src)) return card.src;
  // 数据里未提供扩展名，实际文件为 .jpg
  return `${card.src}.jpg`;
}

function pickRandom(cards) {
  return cards[Math.floor(Math.random() * cards.length)];
}

function createCardElement(card, variant = "") {
  const el = document.createElement("div");
  el.className = `image-card ${variant}`.trim();
  el.style.backgroundImage = `url(${resolveCardImage(card)})`;

  const label = document.createElement("div");
  label.className = "card-label";
  label.textContent = card?.name || "未知卡牌";
  el.appendChild(label);
  return el;
}

function fillReserved(stackEl, cards) {
  if (!stackEl) return;
  stackEl.innerHTML = "";
  for (let i = 0; i < 3; i++) {
    stackEl.appendChild(createCardElement(pickRandom(cards), "reserved"));
  }
}

function fillOwned(ownedEl, cards) {
  if (!ownedEl) return;
  ownedEl.innerHTML = "";
  for (let i = 0; i < 10; i++) {
    ownedEl.appendChild(createCardElement(pickRandom(cards), "owned"));
  }
}

function fillTokens(gridEl) {
  if (!gridEl) return;
  gridEl.innerHTML = "";
  for (let i = 0; i < 8; i++) {
    const t = document.createElement("img");
    t.className = "token";
    t.src = TOKEN_IMAGE;
    t.alt = "token";
    gridEl.appendChild(t);
  }
}

function populatePlayers(cards) {
  PLAYER_AREAS.forEach((selector) => {
    const area = document.querySelector(selector);
    if (!area) return;
    fillReserved(area.querySelector(".reserved-stack"), cards);
    fillTokens(area.querySelector(".token-grid"));
    fillOwned(area.querySelector(".owned-section"), cards);
  });
}

function decoratePreview(el, card, labelText) {
  if (!el) return;
  el.classList.add("has-card");
  el.style.backgroundImage = `url(${resolveCardImage(card)})`;
  el.innerHTML = "";
  if (labelText) {
    const label = document.createElement("span");
    label.className = "deck-label";
    label.textContent = labelText;
    el.appendChild(label);
  }
}

function populateCenter(cards) {
  document.querySelectorAll(".zone .deck").forEach((deck) => {
    const text = deck.textContent.trim();
    decoratePreview(deck, pickRandom(cards), text);
  });
  document.querySelectorAll(".zone .display .slot").forEach((slot) => {
    decoratePreview(slot, pickRandom(cards));
  });
}

function init() {
  loadCardData()
    .then((cards) => {
      if (!cards.length) {
        throw new Error("cards.json 中没有可用的卡牌数据");
      }
      populatePlayers(cards);
      populateCenter(cards);
    })
    .catch((err) => {
      console.error(err);
    });
}

document.addEventListener("DOMContentLoaded", init);
