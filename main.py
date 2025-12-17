from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import List, Optional


@dataclass(frozen=True)
class GameState:
    """Minimal, console-friendly game state."""

    grid_size: int
    player_pos: int
    opponent_pos: int
    current_player: str
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


def get_legal_actions(state: GameState) -> List[str]:
    """Return actions available to the actor whose turn it is."""

    position = state.player_pos if state.current_player == "player" else state.opponent_pos
    actions: List[str] = ["stay"]
    if position > 0:
        actions.append("left")
    if position < state.grid_size - 1:
        actions.append("right")
    return actions


def _move(position: int, action: str) -> int:
    if action == "left":
        return position - 1
    if action == "right":
        return position + 1
    return position


def _check_winner(state: GameState, actor: str, new_pos: int) -> Optional[str]:
    if actor == "player" and new_pos == state.grid_size - 1:
        return actor
    if actor == "opponent" and new_pos == 0:
        return actor

    other_pos = state.opponent_pos if actor == "player" else state.player_pos
    if new_pos == other_pos:
        return actor
    return None


def apply_action(state: GameState, action: str) -> GameState:
    if state.is_terminal:
        return state

    if action not in get_legal_actions(state):
        raise ValueError(f"Illegal action {action!r} for {state.current_player}")

    actor = state.current_player
    new_pos = _move(state.player_pos if actor == "player" else state.opponent_pos, action)
    winner = _check_winner(state, actor, new_pos)

    if actor == "player":
        updated_state = replace(state, player_pos=new_pos)
        next_player = "opponent"
    else:
        updated_state = replace(state, opponent_pos=new_pos)
        next_player = "player"

    if winner:
        return replace(updated_state, winner=winner)

    return replace(updated_state, current_player=next_player)


def random_if_else_ai(state: GameState) -> str:
    """A terrible policy that mostly stumbles toward the goal."""

    actions = get_legal_actions(state)
    position = state.player_pos if state.current_player == "player" else state.opponent_pos
    target = state.grid_size - 1 if state.current_player == "player" else 0

    if random.random() < 0.2:
        return random.choice(actions)

    if position < target and "right" in actions:
        return "right"
    if position > target and "left" in actions:
        return "left"

    return random.choice(actions)


def run_episode(grid_size: int = 7, seed: Optional[int] = None) -> GameState:
    if seed is not None:
        random.seed(seed)

    state = create_initial_state(grid_size)
    while not state.is_terminal:
        action = random_if_else_ai(state)
        state = apply_action(state, action)
    return state


def main() -> None:
    final_state = run_episode()
    print("Final state:")
    print(final_state)
    if final_state.winner:
        print(f"Winner: {final_state.winner}")


if __name__ == "__main__":
    main()
