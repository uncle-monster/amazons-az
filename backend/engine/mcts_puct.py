from __future__ import annotations
import torch
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.engine.board import Board, P1, P2
from backend.engine.nn_policy import nn_value_and_logits
from backend.engine.rules import Action, apply_action, generate_legal_actions, is_terminal
from backend.engine.nn_policy import nn_logits_batched_from_feat
from backend.nn.encode import encode_state

def root_value_from_visits(root: "Node") -> float:
    """
    Compute an MCTS-improved value estimate for the root:
      v = sum_a (N_a / sumN) * Q_a
    This is from the root player's perspective (same as Q values stored on edges).
    """
    if not root.children:
        return 0.0
    Ns = np.asarray([e.N for e in root.children.values()], dtype=np.float32)
    sumN = float(np.sum(Ns))
    if sumN <= 0:
        return 0.0
    Qs = np.asarray([e.Q() for e in root.children.values()], dtype=np.float32)
    v = float(np.sum((Ns / sumN) * Qs))
    return v

def other(player: int) -> int:
    return P2 if player == P1 else P1


def softmax(logits: np.ndarray) -> np.ndarray:
    if logits.size == 0:
        return logits
    m = float(np.max(logits))
    ex = np.exp(logits - m)
    s = float(np.sum(ex))
    if s <= 0:
        # fallback uniform
        return np.ones_like(logits, dtype=np.float32) / float(len(logits))
    return (ex / s).astype(np.float32)

def add_dirichlet_noise(
    priors: np.ndarray,
    alpha: float,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    priors: shape (A,), sums to 1
    returns: mixed priors (A,)
    """
    if priors.size == 0:
        return priors
    noise = rng.dirichlet([alpha] * int(priors.size)).astype(np.float32)
    mixed = (1.0 - float(epsilon)) * priors + float(epsilon) * noise
    # renormalize for numeric safety
    s = float(np.sum(mixed))
    if s > 0:
        mixed = (mixed / s).astype(np.float32)
    return mixed


@dataclass
class Edge:
    P: float
    N: int = 0
    W: float = 0.0
    child: Optional["Node"] = None

    def Q(self) -> float:
        return 0.0 if self.N == 0 else self.W / self.N


@dataclass
class Node:
    board: Board
    player_to_move: int
    parent: Optional["Node"] = None
    parent_action: Optional[Action] = None

    N: int = 0  # state visit count
    expanded: bool = False
    children: Dict[Action, Edge] = field(default_factory=dict)


def puct_score(node: Node, edge: Edge, c_puct: float) -> float:
    # Q + U
    q = edge.Q()
    u = c_puct * edge.P * math.sqrt(max(1, node.N)) / (1 + edge.N)
    return q + u


def select_action_puct(node: Node, c_puct: float) -> Action:
    best_a = None
    best_s = -1e18
    for a, e in node.children.items():
        s = puct_score(node, e, c_puct)
        if s > best_s:
            best_s = s
            best_a = a
    assert best_a is not None
    return best_a


def expand_with_nn(
    node: Node,
    net,
    device: str,
    max_actions_expand: Optional[int],
    rng: random.Random,
) -> float:
    """
    Expands node: fills children edges with prior P(s,a) from NN policy.
    Returns NN value v(s) from perspective of node.player_to_move.
    """
    # Terminal: value = -1 for side to move
    if is_terminal(node.board, node.player_to_move):
        node.expanded = True
        node.children = {}
        return -1.0

    actions = generate_legal_actions(node.board, node.player_to_move)

    net.eval()
    x_board = torch.from_numpy(encode_state(node.board, node.player_to_move)).unsqueeze(0).to(device)  # (1,4,10,10)
    feat = net.forward_features(x_board)  # (1,C,10,10)
    v = net.value(feat).item()

    if len(actions) == 0:
        node.expanded = True
        node.children = {}
        return -1.0

    if max_actions_expand is not None and len(actions) > max_actions_expand:
        # 分批计算所有 actions 的 logits（不会重复 trunk）
        all_logits = nn_logits_batched_from_feat(
            net=net,
            feat=feat,
            actions=actions,
            device=device,
            batch_size=64,   # CPU: 32/64 比较合适
        )
        k = int(max_actions_expand)
        top_idx = np.argpartition(-all_logits, k - 1)[:k]
        top_idx = top_idx[np.argsort(-all_logits[top_idx])]

        actions = [actions[int(i)] for i in top_idx]
        logits = all_logits[top_idx].astype(np.float32)
    else:
        # 不需要 Top-K：直接对 actions 算 logits（也用 batched，统一路径）
        logits = nn_logits_batched_from_feat(
            net=net,
            feat=feat,
            actions=actions,
            device=device,
            batch_size=64,
        )

    priors = softmax(logits)
    node.children = {a: Edge(P=float(p)) for a, p in zip(actions, priors)}
    node.expanded = True
    return float(v)


def backprop(path: List[Tuple[Node, Action]], leaf_value: float) -> None:
    """
    path: list of (node, action_taken_from_node)
    leaf_value: value from perspective of leaf node's player_to_move.
    We will propagate back to root, flipping sign each ply.
    """
    v = leaf_value
    # last node in path is the parent of leaf child (edge taken). We update edges along path.
    for node, action in reversed(path):
        node.N += 1
        edge = node.children[action]
        edge.N += 1
        edge.W += v
        v = -v  # switch player perspective
    # also count root visit if path empty? (not necessary, but keep consistent)
    if not path:
        # means root itself was terminal; no update needed
        pass


def mcts_puct_search(
    board: Board,
    player_to_move: int,
    net,
    device: str = "cpu",
    simulations: int = 200,
    c_puct: float = 1.5,
    max_actions_expand: Optional[int] = 200,
    seed: int = 0,
) -> Action:
    rng = random.Random(seed)
    root = Node(board=board, player_to_move=player_to_move)

    # Ensure root expanded once so children have priors
    _ = expand_with_nn(root, net=net, device=device, max_actions_expand=max_actions_expand, rng=rng)

    if not root.children:
        raise RuntimeError("No legal moves (terminal).")

    for _ in range(simulations):
        node = root
        path: List[Tuple[Node, Action]] = []

        # Selection down the tree
        while node.expanded and node.children:
            a = select_action_puct(node, c_puct=c_puct)
            path.append((node, a))
            edge = node.children[a]

            if edge.child is None:
                # create child by applying action
                next_board = apply_action(node.board, node.player_to_move, a)
                edge.child = Node(
                    board=next_board,
                    player_to_move=other(node.player_to_move),
                    parent=node,
                    parent_action=a,
                )
                node = edge.child
                break
            else:
                node = edge.child

        # Expansion/Evaluation at leaf
        leaf_v = expand_with_nn(node, net=net, device=device, max_actions_expand=max_actions_expand, rng=rng)

        # Backprop along path (leaf_v is from perspective of leaf node.player_to_move)
        backprop(path, leaf_v)

    # Choose action with max visit count
    best_action = max(root.children.items(), key=lambda kv: kv[1].N)[0]
    return best_action

def mcts_puct_policy(
    board: Board,
    player_to_move: int,
    net,
    device: str = "cpu",
    simulations: int = 80,
    c_puct: float = 1.5,
    max_actions_expand: Optional[int] = 200,
    seed: int = 0,
    temperature: float = 1.0,
    dirichlet_alpha: float | None = None,
    dirichlet_epsilon: float = 0.25,
    dirichlet_seed: int | None = None,
) -> Tuple[List[Action], List[float], Action, float, float]:
    """
    Runs PUCT MCTS and returns:
      actions: list of root actions
      pi: visit-count distribution over actions (temperature adjusted)
      chosen_action: sampled from pi (or argmax if temperature ~ 0)
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(dirichlet_seed if dirichlet_seed is not None else seed)

    root = Node(board=board, player_to_move=player_to_move)

    # ---- Root expansion (special: optional Dirichlet noise) ----
    if is_terminal(root.board, root.player_to_move):
        raise RuntimeError("No legal moves (terminal).")

    actions = generate_legal_actions(root.board, root.player_to_move)
    if len(actions) == 0:
        raise RuntimeError("No legal moves (terminal).")

    net.eval()
    x_board = torch.from_numpy(encode_state(root.board, root.player_to_move)).unsqueeze(0).to(device)  # (1,4,10,10)
    feat = net.forward_features(x_board)  # (1,C,10,10)
    v_root = float(net.value(feat).item())

    # logits for ALL actions (batched, no repeated trunk)
    all_logits = nn_logits_batched_from_feat(
        net=net,
        feat=feat,
        actions=actions,
        device=device,
        batch_size=64,   # CPU 建议 32/64；根节点是热点
    )

    # optional Top-K pruning using logits
    if max_actions_expand is not None and len(actions) > max_actions_expand:
        k = int(max_actions_expand)
        top_idx = np.argpartition(-all_logits, k - 1)[:k]
        top_idx = top_idx[np.argsort(-all_logits[top_idx])]
        actions = [actions[int(i)] for i in top_idx]
        logits = all_logits[top_idx].astype(np.float32)
    else:
        logits = all_logits.astype(np.float32)

    priors = softmax(logits)

    if dirichlet_alpha is not None:
        priors = add_dirichlet_noise(
            priors=priors,
            alpha=float(dirichlet_alpha),
            epsilon=float(dirichlet_epsilon),
            rng=np_rng,
        )

    root.children = {a: Edge(P=float(p)) for a, p in zip(actions, priors)}
    root.expanded = True
    if not root.children:
        raise RuntimeError("No legal moves (terminal).")

    for _ in range(simulations):
        node = root
        path: List[Tuple[Node, Action]] = []

        while node.expanded and node.children:
            a = select_action_puct(node, c_puct=c_puct)
            path.append((node, a))
            edge = node.children[a]

            if edge.child is None:
                next_board = apply_action(node.board, node.player_to_move, a)
                edge.child = Node(
                    board=next_board,
                    player_to_move=other(node.player_to_move),
                    parent=node,
                    parent_action=a,
                )
                node = edge.child
                break
            else:
                node = edge.child

        leaf_v = expand_with_nn(node, net=net, device=device, max_actions_expand=max_actions_expand, rng=rng)
        backprop(path, leaf_v)

    actions = list(root.children.keys())
    counts = np.asarray([root.children[a].N for a in actions], dtype=np.float32)
    v_mcts_root = root_value_from_visits(root)

    if temperature <= 1e-8:
        best_i = int(np.argmax(counts))
        pi = [0.0] * len(actions)
        pi[best_i] = 1.0
        return actions, pi, actions[best_i], float(v_root), float(v_mcts_root)
    
    # pi ∝ counts^(1/tau)
    counts_t = np.power(counts, 1.0 / float(temperature))
    s = float(np.sum(counts_t))
    if s <= 0:
        pi_arr = np.ones_like(counts_t) / float(len(counts_t))
    else:
        pi_arr = counts_t / s

    pi = pi_arr.tolist()
    chosen = rng.choices(actions, weights=pi, k=1)[0]
    return actions, pi, chosen, float(v_root), float(v_mcts_root)

