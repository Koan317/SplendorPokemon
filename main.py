from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import List, Optional, Callable, Dict, Tuple


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
# Evaluation (kept for backward compatibility / fallbacks)
# ============================================================

def evaluate_state(state: GameState, perspective: str) -> float:
    """Simple heuristic (legacy)."""
    if state.winner == perspective:
        return 1.0
    if state.winner and state.winner != perspective:
        return -1.0

    if perspective == "player":
        return state.player_pos / (state.grid_size - 1)
    else:
        return (state.grid_size - 1 - state.opponent_pos) / (state.grid_size - 1)


# ============================================================
# Perfect Solver (Win/Lose/Draw) + Depth guidance
# ============================================================

StateKey = Tuple[int, int, str]  # (player_pos, opponent_pos, current_player)

# outcome from the viewpoint of state.current_player:
# +1 = current player can force a win
#  0 = current player can force at least a draw (or game is draw with best play)
# -1 = current player will lose with best opponent play
Outcome = int

# steps:
# - for winning states: minimal steps to force a win (assuming opponent delays)
# - for losing states: maximal steps to delay loss (assuming current tries to delay)
# - for draw states: a soft "progress" metric (optional, used only for tie-breaks)
Steps = int


class SolverCache:
    """Caches solved tables per grid_size."""
    def __init__(self) -> None:
        self._tables: Dict[int, Tuple[Dict[StateKey, Outcome], Dict[StateKey, Steps]]] = {}

    def get(self, grid_size: int) -> Tuple[Dict[StateKey, Outcome], Dict[StateKey, Steps]]:
        if grid_size not in self._tables:
            self._tables[grid_size] = solve_game_tables(grid_size)
        return self._tables[grid_size]


_SOLVER_CACHE = SolverCache()


def solve_game_tables(grid_size: int) -> Tuple[Dict[StateKey, Outcome], Dict[StateKey, Steps]]:
    """
    Solve all reachable non-terminal states for a given grid_size using retrograde analysis.
    Correctly handles cycles => draw.
    """
    # Build all non-terminal states (positions cannot be equal in a non-terminal state)
    states: List[StateKey] = []
    for p in range(grid_size):
        for o in range(grid_size):
            if p == o:
                continue
            states.append((p, o, "player"))
            states.append((p, o, "opponent"))

    # Precompute successors and predecessors
    succ: Dict[StateKey, List[Tuple[Optional[StateKey], bool]]] = {}
    # each successor entry: (next_state_key or None if terminal, is_immediate_win_for_mover)
    pred: Dict[StateKey, List[StateKey]] = {s: [] for s in states}

    # outcome table initialization
    outcome: Dict[StateKey, Outcome] = {s: 0 for s in states}  # 0 means unknown for now (not draw yet)
    known: Dict[StateKey, bool] = {s: False for s in states}

    # remaining moves count for loss propagation
    remaining: Dict[StateKey, int] = {}

    # queue for propagation
    queue: List[StateKey] = []

    for s in states:
        p, o, cp = s
        gs = GameState(grid_size=grid_size, player_pos=p, opponent_pos=o, current_player=cp)
        actions = get_legal_actions(gs)

        s_succ: List[Tuple[Optional[StateKey], bool]] = []
        immediate_win = False

        for a in actions:
            ns = apply_action(gs, a)
            if ns.is_terminal:
                # terminal winner is always the mover in this ruleset
                immediate_win = True
                s_succ.append((None, True))
            else:
                nk: StateKey = (ns.player_pos, ns.opponent_pos, ns.current_player)
                s_succ.append((nk, False))
                pred[nk].append(s)

        succ[s] = s_succ

        if immediate_win:
            # If you have a move that wins immediately, this state is winning.
            outcome[s] = +1
            known[s] = True
            queue.append(s)
        else:
            # Only count non-terminal successors (terminal successors already handled above)
            remaining[s] = sum(1 for (nk, iswin) in s_succ if nk is not None)

    # Retrograde propagation:
    # If a state is losing for the player to move, all predecessors are winning (they can move into it).
    # If a state is winning for player to move, predecessors lose one "hope"; if all successors are winning for next player => predecessor is losing.
    while queue:
        s = queue.pop(0)
        s_val = outcome[s]
        for ps in pred[s]:
            if known[ps]:
                continue

            if s_val == -1:
                # predecessor can move to a losing state for the opponent -> predecessor is winning
                outcome[ps] = +1
                known[ps] = True
                queue.append(ps)
            elif s_val == +1:
                # predecessor moved into a winning state for the opponent, reduces options
                remaining[ps] -= 1
                if remaining[ps] <= 0:
                    outcome[ps] = -1
                    known[ps] = True
                    queue.append(ps)

    # Any still-unknown state is draw (0)
    for s in states:
        if not known[s]:
            outcome[s] = 0

    # Now compute step guidance tables for better move selection:
    # - win_steps: minimal steps to force a win (opponent delays)
    # - lose_steps: maximal steps to delay loss
    # For draws, we use a mild heuristic (progress-to-goal) to break ties.
    steps: Dict[StateKey, Steps] = {}

    # Initialize steps for known terminal-adjacent wins:
    # If state is winning and has an immediate win move => steps = 1
    for s in states:
        if outcome[s] != +1:
            continue
        if any(iswin for (_nk, iswin) in succ[s]):
            steps[s] = 1

    # Iteratively relax until stable (state space is tiny)
    changed = True
    for _ in range(2000):
        if not changed:
            break
        changed = False
        for s in states:
            val = outcome[s]
            p, o, cp = s

            if val == +1:
                # Winning: choose a move to a losing state for opponent, minimizing time-to-win
                candidates: List[int] = []
                gs = GameState(grid_size=grid_size, player_pos=p, opponent_pos=o, current_player=cp)
                for a in get_legal_actions(gs):
                    ns = apply_action(gs, a)
                    if ns.is_terminal:
                        candidates.append(1)
                    else:
                        nk = (ns.player_pos, ns.opponent_pos, ns.current_player)
                        if outcome[nk] == -1:
                            # opponent is losing at nk; they will try to delay our win
                            # so we take 1 + (their best delay), which is steps[nk] if computed as lose-steps
                            # if not available yet, skip for now
                            if nk in steps:
                                candidates.append(1 + steps[nk])

                if candidates:
                    newv = min(candidates)
                    if steps.get(s) != newv:
                        steps[s] = newv
                        changed = True

            elif val == -1:
                # Losing: choose a move that maximizes time-to-loss (delay defeat)
                candidates = []
                gs = GameState(grid_size=grid_size, player_pos=p, opponent_pos=o, current_player=cp)
                for a in get_legal_actions(gs):
                    ns = apply_action(gs, a)
                    if ns.is_terminal:
                        # immediate loss for mover would be terminal for mover? In this ruleset, mover wins terminal,
                        # so terminal here cannot be loss for mover. keep safe:
                        continue
                    nk = (ns.player_pos, ns.opponent_pos, ns.current_player)
                    # from nk, opponent to move; if nk is winning for them, then our loss is coming.
                    if outcome[nk] == +1:
                        if nk in steps:
                            candidates.append(1 + steps[nk])

                if candidates:
                    newv = max(candidates)
                    if steps.get(s) != newv:
                        steps[s] = newv
                        changed = True

            else:
                # Draw: store a small heuristic score for tie-break (not a "steps to outcome")
                # Higher is better for current player: prefer progress towards own goal and away from opponent goal.
                # This is only for selecting among draw-preserving moves.
                if s not in steps:
                    # normalize into small int range
                    if cp == "player":
                        prog = p
                        opp_prog = (grid_size - 1 - o)
                    else:
                        prog = (grid_size - 1 - o)
                        opp_prog = p
                    steps[s] = int((prog - opp_prog) * 10)

    # For any missing step value in win/lose states (rare), fallback to 0
    for s in states:
        if s not in steps:
            steps[s] = 0

    return outcome, steps


def _rank_actions_for_state(state: GameState) -> List[Tuple[str, Outcome, int]]:
    """
    Return list of (action, resulting_outcome_for_current_player, guidance_steps)
    sorted best-first for current_player.
    """
    outcome_table, steps_table = _SOLVER_CACHE.get(state.grid_size)
    me = state.current_player

    scored: List[Tuple[str, Outcome, int]] = []
    for a in get_legal_actions(state):
        ns = apply_action(state, a)
        if ns.is_terminal:
            # mover wins
            scored.append((a, +1, 1))
            continue

        nk: StateKey = (ns.player_pos, ns.opponent_pos, ns.current_player)
        # outcome_table[nk] is from viewpoint of ns.current_player (the opponent of 'me')
        # so from 'me' viewpoint it's negated
        o_for_me: Outcome = -outcome_table[nk]
        # guidance:
        # for winning: prefer smaller steps
        # for losing: prefer larger steps (delay)
        # for draw: use steps_table directly
        guide = steps_table[nk]
        scored.append((a, o_for_me, guide))

    # Sort rule:
    # 1) higher outcome (+1 > 0 > -1)
    # 2) if outcome == +1: smaller guide better
    # 3) if outcome == -1: larger guide better
    # 4) if outcome == 0: larger guide better (heuristic)
    def key_fn(item: Tuple[str, Outcome, int]) -> Tuple[int, int]:
        _, o, g = item
        primary = o
        if o == +1:
            secondary = -g  # smaller steps -> larger -g
        else:
            secondary = g   # draw prefer higher heuristic; lose prefer larger delay
        return (primary, secondary)

    scored.sort(key=key_fn, reverse=True)
    return scored


def perfect_policy(state: GameState) -> str:
    """Best play (win if possible, else draw if possible, else delay)."""
    ranked = _rank_actions_for_state(state)
    return ranked[0][0]


def strong_humanlike_policy(state: GameState, temperature: float, rng: random.Random) -> str:
    """
    Strong play with controlled variability:
    - Never intentionally choose a losing move if a win/draw is available.
    - Within the best outcome class, sample using a soft preference (temperature).
    This makes the AI feel less robotic while staying very strong.
    """
    ranked = _rank_actions_for_state(state)
    if not ranked:
        return random.choice(get_legal_actions(state))

    # Partition by best outcome
    best_outcome = ranked[0][1]
    pool = [x for x in ranked if x[1] == best_outcome]

    if len(pool) == 1:
        return pool[0][0]

    # Convert (outcome, guide) to weights within the pool
    # For wins: prefer smaller guide (faster win)
    # For draws: prefer larger guide (better position)
    # For losses: prefer larger guide (delay)
    scores = []
    for (_a, o, g) in pool:
        if o == +1:
            # invert: smaller g => larger score
            s = -g
        else:
            s = g
        scores.append(s)

    # Softmax with temperature (higher temperature => more randomness)
    t = max(1e-6, temperature)
    m = max(scores)
    exps = [pow(2.718281828, (s - m) / t) for s in scores]
    total = sum(exps)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(exps):
        acc += w
        if acc >= r:
            return pool[i][0]
    return pool[-1][0]


# ============================================================
# AI Policies (public API compatible)
# ============================================================

def random_policy(state: GameState) -> str:
    """Baseline random policy."""
    return random.choice(get_legal_actions(state))


def greedy_depth2_policy(state: GameState) -> str:
    """Legacy policy (kept as fallback / easiest tiers)."""
    me = state.current_player
    best_score = -1e9
    best_action = None

    for action in get_legal_actions(state):
        s1 = apply_action(state, action)

        if s1.is_terminal:
            score = evaluate_state(s1, me)
        else:
            score = min(
                evaluate_state(apply_action(s1, reply), me)
                for reply in get_legal_actions(s1)
            )

        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def ai_with_difficulty(state: GameState, level: int) -> Optional[str]:
    """
    Difficulty-controlled AI with an option to turn it off.

    level: -1 (disabled) | 0 (easiest) ~ 4 (hardest)

    Guarantee goal for level=2:
    - Strong enough to be competitive with a rules-savvy human (roughly parity),
      by playing draw/win-safe and sampling among best lines with mild variability.
    """
    if level == -1:
        return None

    if not 0 <= level <= 4:
        raise ValueError("Difficulty level must be between 0 and 4")

    # Use a deterministic RNG seeded by the current state to keep "fair but stable" behavior:
    # This avoids feeling like coin-flip cheating while still adding variability.
    seed = (state.grid_size * 1000
            + state.player_pos * 100
            + state.opponent_pos * 10
            + (0 if state.current_player == "player" else 1))
    rng = random.Random(seed)

    if level == 0:
        # Very weak: mostly random
        if rng.random() < 0.85:
            return random_policy(state)
        return greedy_depth2_policy(state)

    if level == 1:
        # Weak: greedy with frequent blunders
        if rng.random() < 0.35:
            return random_policy(state)
        return greedy_depth2_policy(state)

    if level == 2:
        # Standard (target): strong, win/draw-safe, mildly non-robotic
        # Temperature tuned so it doesn't always pick the same perfect line,
        # but it will not throw away a win or a draw.
        return strong_humanlike_policy(state, temperature=1.2, rng=rng)

    if level == 3:
        # Hard: near-perfect, low randomness
        return strong_humanlike_policy(state, temperature=0.35, rng=rng)

    # level == 4 (hardest): perfect/solved
    return perfect_policy(state)


# ============================================================
# Match Runner (for testing / benchmarking)
# ============================================================

def run_match(
    ai_player: Callable[[GameState], str],
    ai_opponent: Callable[[GameState], str],
    games: int = 50,
) -> dict:
    """Run AI vs AI matches and return win statistics."""
    results = {"player": 0, "opponent": 0}

    for seed in range(games):
        random.seed(seed)
        state = create_initial_state()

        # Safety cap: prevent infinite loops in draw-ish lines
        # (solver will treat those as drawable, but runner must terminate)
        max_plies = 500

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
            # Treat non-terminated within cap as a draw; count half-half to avoid bias.
            # Keep results dict compatible (no new keys).
            results["player"] += 0
            results["opponent"] += 0

    return results


# ============================================================
# Entry Point
# ============================================================

def main() -> None:
    print("Testing difficulty levels (AI as player vs AI-random opponent):")
    for level in range(5):
        result = run_match(
            lambda s, lv=level: ai_with_difficulty(s, lv) or "stay",
            random_policy,
            games=200,
        )
        print(f"AI level {level}: {result}")

    print("\nQuick sanity: level 4 vs level 4 (should be stable / mostly draws depending on solvability):")
    result = run_match(
        lambda s: ai_with_difficulty(s, 4) or "stay",
        lambda s: ai_with_difficulty(s, 4) or "stay",
        games=50,
    )
    print(f"Level4 vs Level4: {result}")


if __name__ == "__main__":
    main()
