from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import List, Optional, Callable, Dict, Tuple
from collections import deque, defaultdict


# ============================================================
# Game State
# ============================================================

@dataclass(frozen=True)
class GameState:
    """Pure game state, no UI, no side effects."""
    grid_size: int
    player_pos: int
    opponent_pos: int
    current_player: str  # "player" | "opponent"
    winner: Optional[str] = None

    @property
    def is_terminal(self) -> bool:
        return self.winner is not None


def create_initial_state(grid_size: int = 7) -> GameState:
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")

    return GameState(
        grid_size=grid_size,
        player_pos=0,
        opponent_pos=grid_size - 1,
        current_player="player",
    )


# ============================================================
# Rules
# ============================================================

def get_legal_actions(state: GameState) -> List[str]:
    """Return all legal actions for current player."""
    pos = state.player_pos if state.current_player == "player" else state.opponent_pos
    actions = ["stay"]

    if pos > 0:
        actions.append("left")
    if pos < state.grid_size - 1:
        actions.append("right")

    return actions


def apply_action(state: GameState, action: str) -> GameState:
    """Apply action and return new GameState."""
    if state.is_terminal:
        return state

    if action not in get_legal_actions(state):
        raise ValueError(f"Illegal action: {action}")

    actor = state.current_player
    pos = state.player_pos if actor == "player" else state.opponent_pos

    new_pos = _move(pos, action)
    winner = _check_winner(state, actor, new_pos)

    if actor == "player":
        next_state = replace(state, player_pos=new_pos)
        next_player = "opponent"
    else:
        next_state = replace(state, opponent_pos=new_pos)
        next_player = "player"

    if winner:
        return replace(next_state, winner=winner)

    return replace(next_state, current_player=next_player)


def _move(position: int, action: str) -> int:
    if action == "left":
        return position - 1
    if action == "right":
        return position + 1
    return position


def _check_winner(state: GameState, actor: str, new_pos: int) -> Optional[str]:
    # Reach goal
    if actor == "player" and new_pos == state.grid_size - 1:
        return actor
    if actor == "opponent" and new_pos == 0:
        return actor

    # Collision
    other_pos = state.opponent_pos if actor == "player" else state.player_pos
    if new_pos == other_pos:
        return actor

    return None


# ============================================================
# Heuristic (used only in weaker AIs / tie-breaks)
# ============================================================

def evaluate_state(state: GameState, perspective: str) -> float:
    """
    A slightly smarter heuristic than the original:
    - progress to goal
    - distance to opponent (collision threats)
    - mild preference for having tempo (being able to threaten collision next)
    """
    if state.winner == perspective:
        return 1.0
    if state.winner and state.winner != perspective:
        return -1.0

    n = state.grid_size - 1
    if perspective == "player":
        my = state.player_pos
        op = state.opponent_pos
        my_prog = my / n
        op_prog = (n - op) / n
    else:
        my = state.opponent_pos
        op = state.player_pos
        my_prog = (n - my) / n
        op_prog = op / n

    dist = abs(state.player_pos - state.opponent_pos)
    # closer increases tactical volatility; prefer being the one who is closer to scoring
    score = (my_prog - 0.85 * op_prog)
    score += 0.12 * (1.0 / max(1, dist))  # tactical value

    return max(-1.0, min(1.0, score))


# ============================================================
# Perfect Solver (Retrograde analysis: Win/Lose/Draw with cycles)
# ============================================================

StateKey = Tuple[int, int, int]  # (player_pos, opponent_pos, turn) where turn: 0=player, 1=opponent
Outcome = int  # +1 win for current mover, 0 draw, -1 loss for current mover


def _turn_id(current_player: str) -> int:
    return 0 if current_player == "player" else 1


def _player_of_turn(turn: int) -> str:
    return "player" if turn == 0 else "opponent"


class SolverCache:
    def __init__(self) -> None:
        self._tables: Dict[int, Tuple[Dict[StateKey, Outcome], Dict[StateKey, int]]] = {}

    def get(self, grid_size: int) -> Tuple[Dict[StateKey, Outcome], Dict[StateKey, int]]:
        if grid_size not in self._tables:
            self._tables[grid_size] = _solve_tables(grid_size)
        return self._tables[grid_size]


_SOLVER = SolverCache()


def _solve_tables(grid_size: int) -> Tuple[Dict[StateKey, Outcome], Dict[StateKey, int]]:
    """
    Solve all non-terminal states for a given grid_size.

    Produces:
    - outcome[key]: +1 / 0 / -1 from viewpoint of side-to-move in that key
    - dist[key]: for winning states: minimal plies to force win
                for losing states: maximal plies to delay loss
                for draw states: a pressure heuristic (bigger=better for side-to-move)
    """
    keys: List[StateKey] = []
    for p in range(grid_size):
        for o in range(grid_size):
            if p == o:
                continue
            keys.append((p, o, 0))
            keys.append((p, o, 1))

    # Build successor lists and predecessor lists
    succ: Dict[StateKey, List[Tuple[Optional[StateKey], bool]]] = {}
    pred: Dict[StateKey, List[StateKey]] = {k: [] for k in keys}
    outdeg: Dict[StateKey, int] = {}

    # Retrograde structures
    outcome: Dict[StateKey, Outcome] = {k: 0 for k in keys}
    known: Dict[StateKey, bool] = {k: False for k in keys}
    remaining: Dict[StateKey, int] = {}

    q = deque()

    for k in keys:
        p, o, t = k
        gs = GameState(grid_size=grid_size, player_pos=p, opponent_pos=o, current_player=_player_of_turn(t))
        actions = get_legal_actions(gs)

        s_list: List[Tuple[Optional[StateKey], bool]] = []
        immediate_win = False

        for a in actions:
            ns = apply_action(gs, a)
            if ns.is_terminal:
                immediate_win = True
                s_list.append((None, True))
            else:
                nk = (ns.player_pos, ns.opponent_pos, _turn_id(ns.current_player))
                s_list.append((nk, False))
                pred[nk].append(k)

        succ[k] = s_list
        outdeg[k] = sum(1 for nk, _ in s_list if nk is not None)

        if immediate_win:
            outcome[k] = +1
            known[k] = True
            q.append(k)
        else:
            remaining[k] = outdeg[k]

    # Retrograde propagation
    while q:
        k = q.popleft()
        k_val = outcome[k]

        for pk in pred[k]:
            if known[pk]:
                continue

            if k_val == -1:
                # if you can move to a state where opponent-to-move is losing => you are winning
                outcome[pk] = +1
                known[pk] = True
                q.append(pk)
            elif k_val == +1:
                # this successor is winning for opponent; reduce remaining "hope"
                remaining[pk] -= 1
                if remaining[pk] <= 0:
                    outcome[pk] = -1
                    known[pk] = True
                    q.append(pk)

    # Unknown => draw
    for k in keys:
        if not known[k]:
            outcome[k] = 0

    # Distance/pressure guidance
    dist: Dict[StateKey, int] = {}

    # initialize trivial win distance: if immediate win exists => 1
    for k in keys:
        if outcome[k] != +1:
            continue
        if any(iswin for (_nk, iswin) in succ[k]):
            dist[k] = 1

    # iterative relaxation (tiny state space)
    for _ in range(2000):
        changed = False
        for k in keys:
            val = outcome[k]
            if val == +1:
                # choose move to a losing state for opponent with minimal steps
                cands = []
                for nk, iswin in succ[k]:
                    if iswin:
                        cands.append(1)
                        continue
                    if nk is None:
                        continue
                    if outcome[nk] == -1 and nk in dist:
                        cands.append(1 + dist[nk])
                if cands:
                    newv = min(cands)
                    if dist.get(k) != newv:
                        dist[k] = newv
                        changed = True

            elif val == -1:
                # choose move to opponent-winning state, but maximize steps to delay loss
                cands = []
                for nk, iswin in succ[k]:
                    if iswin:
                        # cannot be immediate loss because mover wins terminal under these rules
                        continue
                    if nk is None:
                        continue
                    if outcome[nk] == +1 and nk in dist:
                        cands.append(1 + dist[nk])
                if cands:
                    newv = max(cands)
                    if dist.get(k) != newv:
                        dist[k] = newv
                        changed = True

            else:
                # draw: pressure heuristic for tie-break
                if k not in dist:
                    p, o, t = k
                    # for side-to-move, progress vs opponent progress
                    if t == 0:  # player to move
                        my_prog = p
                        op_prog = (grid_size - 1 - o)
                    else:
                        my_prog = (grid_size - 1 - o)
                        op_prog = p
                    d = abs(p - o)
                    # bigger is better: more progress, closer tactical tension, and being "ahead"
                    dist[k] = int((my_prog - op_prog) * 20 - d * 2)
        if not changed:
            break

    for k in keys:
        if k not in dist:
            dist[k] = 0

    return outcome, dist


# ============================================================
# Move ranking (the "personality" of strong AIs is here)
# ============================================================

def _state_key_from_state(s: GameState) -> StateKey:
    return (s.player_pos, s.opponent_pos, _turn_id(s.current_player))


def _opponent_branching(s: GameState) -> int:
    """How many legal replies the next player has."""
    return len(get_legal_actions(s))


def _pressure_metric(s: GameState, me: str) -> int:
    """
    Higher = feels more 'dominant':
    - reduce opponent branching
    - keep tactical tension (distance small but not suicidal)
    - keep opponent away from their goal
    """
    n = s.grid_size - 1
    if me == "player":
        my = s.player_pos
        op = s.opponent_pos
        my_to_goal = n - my
        op_to_goal = op
    else:
        my = s.opponent_pos
        op = s.player_pos
        my_to_goal = my
        op_to_goal = n - op

    d = abs(s.player_pos - s.opponent_pos)
    # branching reduction: fewer choices for opponent feels oppressive
    br = _opponent_branching(s)
    # pressure likes: opponent far from goal, me closer; tactical distance small-ish
    return int((op_to_goal - my_to_goal) * 10 - br * 6 - d * 2)


def _rank_actions(
    state: GameState,
    style: str,
    rng: random.Random,
) -> List[Tuple[str, int, int, int]]:
    """
    Returns list of tuples:
        (action, outcome_for_me, dist_score, pressure_score)
    sorted best-first according to style.

    outcome_for_me: +1 win / 0 draw / -1 loss from the viewpoint of state.current_player
    dist_score: for win => smaller is better (faster mate)
               for loss => larger is better (delay)
               for draw => larger is better (positional pressure heuristic)
    pressure_score: larger is more oppressive / constricting
    """
    outcome_table, dist_table = _SOLVER.get(state.grid_size)
    me = state.current_player
    base_key = _state_key_from_state(state)

    ranked = []

    for a in get_legal_actions(state):
        ns = apply_action(state, a)
        if ns.is_terminal:
            ranked.append((a, +1, 1, 10_000))
            continue

        nk = _state_key_from_state(ns)
        # outcome_table[nk] is from viewpoint of ns.current_player (opponent of me),
        # so for me it's negated:
        out_for_me = -outcome_table[nk]

        # dist guidance
        g = dist_table[nk]

        # pressure
        pscore = _pressure_metric(ns, me)

        ranked.append((a, out_for_me, g, pscore))

    # Sorting styles
    def sort_key(item: Tuple[str, int, int, int]) -> Tuple[int, int, int]:
        _a, out, g, ps = item

        # Primary: outcome
        primary = out

        if style == "dominator":
            # Win fast, and among equal outcomes prefer higher pressure (constrict replies).
            if out == +1:
                # faster win first, then pressure
                return (primary, -g, ps)
            if out == 0:
                # draw: maximize pressure, then g
                return (primary, ps, g)
            # losing: maximize delay, and try to keep pressure traps
            return (primary, g, ps)

        if style == "pragmatic":
            # Strong humanlike: preserve best outcome; in draws prefer *complex* lines:
            # more branching for opponent can increase their chance to err.
            # We'll encode that indirectly by preferring slightly lower pressure in draws.
            if out == +1:
                return (primary, -g, ps)
            if out == 0:
                # draw: prefer lines with LOWER pressure (more freedom = more chances for mistakes)
                return (primary, -ps, g)
            return (primary, g, ps)

        if style == "simple":
            # Simple: basically outcome then heuristic only
            if out == +1:
                return (primary, -g, 0)
            if out == 0:
                return (primary, g, 0)
            return (primary, g, 0)

        # default
        if out == +1:
            return (primary, -g, ps)
        if out == 0:
            return (primary, g, ps)
        return (primary, g, ps)

    ranked.sort(key=sort_key, reverse=True)
    return ranked


def _soft_pick(
    candidates: List[Tuple[str, int, int, int]],
    outcome_preference: int,
    temperature: float,
    rng: random.Random,
    draw_prefers_complexity: bool,
) -> str:
    """
    Pick among candidates of the best outcome class with a soft preference.
    temperature high => more variety; low => more deterministic.

    draw_prefers_complexity:
        - True: among draws, prefer lower pressure / higher branching (more human mistakes)
        - False: among draws, prefer higher pressure (more constricting)
    """
    if not candidates:
        raise ValueError("No candidates to pick from")

    best_outcome = candidates[0][1]
    pool = [c for c in candidates if c[1] == best_outcome]

    if len(pool) == 1:
        return pool[0][0]

    t = max(1e-6, temperature)

    # Convert each item to a scalar score
    scores = []
    for (_a, out, g, ps) in pool:
        if out == +1:
            # faster win better; add pressure a bit
            s = (-g * 3.0) + (ps * 0.15)
        elif out == 0:
            # draw: either prefer complexity (lower pressure) or dominance (higher pressure)
            s = ((-ps) if draw_prefers_complexity else ps) * 0.8 + g * 0.15
        else:
            # losing: delay loss
            s = g * 1.2 + ps * 0.05
        scores.append(s)

    m = max(scores)
    weights = [pow(2.718281828, (s - m) / t) for s in scores]
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if acc >= r:
            return pool[i][0]
    return pool[-1][0]


# ============================================================
# AI Policies (difficulty personas)
# ============================================================

def random_policy(state: GameState) -> str:
    return random.choice(get_legal_actions(state))


def weak_greedy_policy(state: GameState, rng: random.Random) -> str:
    """
    Very human-beginner-like:
    - loves moving toward goal
    - often ignores tactical collision threats
    """
    me = state.current_player
    actions = get_legal_actions(state)
    scored = []
    for a in actions:
        ns = apply_action(state, a)
        s = evaluate_state(ns, me)
        # add noise and goal-hunger bias
        s += rng.uniform(-0.25, 0.25)
        scored.append((s, a))
    scored.sort(reverse=True)
    return scored[0][1]


def depth_limited_minimax(state: GameState, depth: int, rng: random.Random) -> str:
    """
    Depth-limited minimax with heuristic eval. Handles cycles by depth cutoff only (weaker than solver).
    Good for mid-low difficulties.
    """
    me = state.current_player

    def rec(s: GameState, d: int) -> float:
        if s.is_terminal:
            return 1.0 if s.winner == me else -1.0
        if d <= 0:
            return evaluate_state(s, me)

        acts = get_legal_actions(s)
        if s.current_player == me:
            best = -1e9
            for a in acts:
                best = max(best, rec(apply_action(s, a), d - 1))
            return best
        else:
            worst = 1e9
            for a in acts:
                worst = min(worst, rec(apply_action(s, a), d - 1))
            return worst

    acts = get_legal_actions(state)
    scored = []
    for a in acts:
        v = rec(apply_action(state, a), depth - 1)
        # tiny noise to avoid robotic ties at low levels
        v += rng.uniform(-0.02, 0.02)
        scored.append((v, a))
    scored.sort(reverse=True)
    return scored[0][1]


def ai_with_difficulty(state: GameState, level: int) -> Optional[str]:
    """
    level: -1 (disabled) | 0..4

    Design goals:
    - level 0: beginner (obvious mistakes)
    - level 1: novice (some lookahead, still blunders)
    - level 2: standard (strong, parity vs strong human)
    - level 3: hard (near-perfect, punishes)
    - level 4: dominator (perfect-ish + oppressive line selection)
    """
    if level == -1:
        return None
    if not 0 <= level <= 4:
        raise ValueError("Difficulty level must be between 0 and 4")

    # Deterministic-ish per-position RNG (feels fair; no "cheating dice" across runs)
    seed = (
        state.grid_size * 1000003
        + state.player_pos * 1009
        + state.opponent_pos * 917
        + (0 if state.current_player == "player" else 1) * 131
        + level * 271
    )
    rng = random.Random(seed)

    # Difficulty 0: mostly random + greedy toward goal
    if level == 0:
        if rng.random() < 0.65:
            return random_policy(state)
        return weak_greedy_policy(state, rng)

    # Difficulty 1: shallow minimax but still blunders sometimes
    if level == 1:
        if rng.random() < 0.18:
            return random_policy(state)
        if rng.random() < 0.25:
            return weak_greedy_policy(state, rng)
        return depth_limited_minimax(state, depth=3, rng=rng)

    # Difficulty 2: strong "humanlike killer"
    # - NEVER throws away a win or a draw if it has it (uses solved outcomes),
    # - BUT among drawable lines, prefers complexity (more chances for YOU to slip).
    if level == 2:
        ranked = _rank_actions(state, style="pragmatic", rng=rng)
        # mild variety; strong but not robotic
        return _soft_pick(
            ranked,
            outcome_preference=+1,
            temperature=1.05,
            rng=rng,
            draw_prefers_complexity=True,
        )

    # Difficulty 3: near-perfect, low randomness, prefers dominance even in draws
    if level == 3:
        ranked = _rank_actions(state, style="dominator", rng=rng)
        return _soft_pick(
            ranked,
            outcome_preference=+1,
            temperature=0.25,
            rng=rng,
            draw_prefers_complexity=False,
        )

    # Difficulty 4: "Dominator" (oppressive perfect play)
    # - deterministic best line selection
    # - fastest wins, strongest constriction in draws
    ranked = _rank_actions(state, style="dominator", rng=rng)
    return ranked[0][0]


# ============================================================
# Match Runner (for testing / benchmarking)
# ============================================================

def run_match(
    ai_player: Callable[[GameState], str],
    ai_opponent: Callable[[GameState], str],
    games: int = 50,
) -> dict:
    results = {"player": 0, "opponent": 0, "draw": 0}

    for seed in range(games):
        random.seed(seed)
        state = create_initial_state()

        # cap to avoid infinite loops in drawable positions
        max_plies = 600
        plies = 0

        while not state.is_terminal and plies < max_plies:
            if state.current_player == "player":
                action = ai_player(state)
            else:
                action = ai_opponent(state)
            state = apply_action(state, action)
            plies += 1

        if state.is_terminal:
            results[state.winner] += 1
        else:
            results["draw"] += 1

    return results


# ============================================================
# Entry Point
# ============================================================

def main() -> None:
    print("AI vs Random baseline (200 games each):")
    for level in range(5):
        r = run_match(
            lambda s, lv=level: ai_with_difficulty(s, lv) or "stay",
            random_policy,
            games=200,
        )
        print(f"Level {level}: {r}")

    print("\nHardness ladder (AI level X as player vs AI level X-1 as opponent):")
    for level in range(1, 5):
        r = run_match(
            lambda s, lv=level: ai_with_difficulty(s, lv) or "stay",
            lambda s, lv=level - 1: ai_with_difficulty(s, lv) or "stay",
            games=200,
        )
        print(f"Level {level} vs Level {level-1}: {r}")

    print("\nTop-tier sanity (Level4 vs Level4):")
    r = run_match(
        lambda s: ai_with_difficulty(s, 4) or "stay",
        lambda s: ai_with_difficulty(s, 4) or "stay",
        games=80,
    )
    print(f"Level4 vs Level4: {r}")


if __name__ == "__main__":
    main()
