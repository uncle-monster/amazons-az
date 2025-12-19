from __future__ import annotations
try:
    from backend.engine.nn_policy import nn_value_and_logits
except Exception:
    nn_value_and_logits = None
from typing import Any
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backend.engine.board import Board, P1, P2
from backend.engine.rules import Action, apply_action, generate_legal_actions, is_terminal


def other(player: int) -> int:
    return P2 if player == P1 else P1


def mobility(board: Board, player: int) -> int:
    return len(generate_legal_actions(board, player))


def evaluate(board: Board, player_to_move: int) -> float:
    if is_terminal(board, player_to_move):
        return -1.0
    my_m = mobility(board, player_to_move)
    opp_m = mobility(board, other(player_to_move))
    return math.tanh((my_m - opp_m) / 200.0)


@dataclass
class Node:
    board: Board
    player_to_move: int
    parent: Optional["Node"] = None
    parent_action: Optional[Action] = None

    N: int = 0
    W: float = 0.0
    children: Dict[Action, "Node"] = field(default_factory=dict)
    untried_actions: Optional[List[Action]] = None

    def Q(self) -> float:
        return 0.0 if self.N == 0 else self.W / self.N

    def is_fully_expanded(self) -> bool:
        return self.untried_actions is not None and len(self.untried_actions) == 0


def uct_score(parent_N: int, child: Node, c: float = 1.4) -> float:
    if child.N == 0:
        return float("inf")
    return child.Q() + c * math.sqrt(math.log(parent_N) / child.N)


def select_child_uct(node: Node, c: float) -> Tuple[Action, Node]:
    best_a, best_child, best_s = None, None, -1e9
    for a, ch in node.children.items():
        s = uct_score(node.N, ch, c=c)
        if s > best_s:
            best_s = s
            best_a, best_child = a, ch
    assert best_a is not None and best_child is not None
    return best_a, best_child


def expand(node: Node, max_actions_expand: Optional[int], rng: random.Random) -> Node:
    if node.untried_actions is None:
        actions = generate_legal_actions(node.board, node.player_to_move)
        if max_actions_expand is not None and len(actions) > max_actions_expand:
            actions = rng.sample(actions, k=max_actions_expand)
        node.untried_actions = actions

    if not node.untried_actions:
        return node

    a = node.untried_actions.pop()
    next_board = apply_action(node.board, node.player_to_move, a)
    next_player = other(node.player_to_move)
    child = Node(board=next_board, player_to_move=next_player, parent=node, parent_action=a)
    node.children[a] = child
    return child


def backprop(node: Node, leaf_value_from_leaf_perspective: float) -> None:
    v = leaf_value_from_leaf_perspective
    cur = node
    while cur is not None:
        cur.N += 1
        cur.W += v
        v = -v
        cur = cur.parent


def _run_mcts_root(
    board: Board,
    player_to_move: int,
    simulations: int,
    c_uct: float,
    max_actions_expand: Optional[int],
    seed: int,
    net: Any = None,          # AmazonsNet or None
    device: str = "cpu",
) -> Node:
    rng = random.Random(seed)
    root = Node(board=board, player_to_move=player_to_move)

    actions = generate_legal_actions(board, player_to_move)

    # If NN is available, rank actions by NN policy logits and take top-K
    if net is not None and nn_value_and_logits is not None and len(actions) > 0 and max_actions_expand is not None:
        _, logits = nn_value_and_logits(net, board, player_to_move, actions, device=device)
        idx = list(range(len(actions)))
        idx.sort(key=lambda i: float(logits[i]), reverse=True)
        idx = idx[:max_actions_expand]
        root.untried_actions = [actions[i] for i in idx]
    else:
        root.untried_actions = actions
        if max_actions_expand is not None and len(root.untried_actions) > max_actions_expand:
            root.untried_actions = rng.sample(root.untried_actions, k=max_actions_expand)

    for _ in range(simulations):
        node = root

        while node.children and node.is_fully_expanded():
            _, node = select_child_uct(node, c=c_uct)

        if not is_terminal(node.board, node.player_to_move):
            node = expand(node, max_actions_expand=max_actions_expand, rng=rng)

        # Leaf evaluation: NN value if provided, else heuristic evaluate()
        if net is not None and nn_value_and_logits is not None:
            v, _ = nn_value_and_logits(net, node.board, node.player_to_move, [], device=device)
            leaf_v = float(v)
        else:
            leaf_v = evaluate(node.board, node.player_to_move)

        backprop(node, leaf_v)

    return root


def mcts_search(
    board: Board,
    player_to_move: int,
    simulations: int = 200,
    c_uct: float = 1.4,
    max_actions_expand: Optional[int] = 300,
    seed: int = 0,
    net = None,
    device: str="cpu",
) -> Action:
    root = _run_mcts_root(board, player_to_move, simulations, c_uct, max_actions_expand, seed, net=net, device=device)
    if not root.children:
        raise RuntimeError("No legal moves from root.")
    best_action = max(root.children.items(), key=lambda kv: kv[1].N)[0]
    return best_action


def mcts_policy(
    board: Board,
    player_to_move: int,
    simulations: int = 200,
    c_uct: float = 1.4,
    max_actions_expand: Optional[int] = 300,
    seed: int = 0,
    temperature: float = 1.0,
    net = None,
    device: str="cpu",
) -> Tuple[List[Action], List[float], Action]:
    """
    Returns:
      actions: list of actions at root (only expanded subset if max_actions_expand used)
      pi: normalized distribution derived from visit counts
      chosen_action: sampled from pi (temperature controls exploration)
    """
    root = _run_mcts_root(board, player_to_move, simulations, c_uct, max_actions_expand, seed, net=net, device=device)
    if not root.children:
        raise RuntimeError("No legal moves from root.")

    actions = list(root.children.keys())
    counts = [root.children[a].N for a in actions]

    # temperature transform
    if temperature <= 1e-8:
        # pick argmax deterministically
        best_i = max(range(len(actions)), key=lambda i: counts[i])
        pi = [0.0] * len(actions)
        pi[best_i] = 1.0
        return actions, pi, actions[best_i]

    counts_t = [c ** (1.0 / temperature) for c in counts]
    s = sum(counts_t)
    pi = [c / s for c in counts_t]

    rng = random.Random(seed + 99991)
    chosen = rng.choices(actions, weights=pi, k=1)[0]
    return actions, pi, chosen