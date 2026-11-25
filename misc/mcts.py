"""
misc.mcts
---------
Early standalone PUCT Monte Carlo Tree Search prototype with random-uniform
evaluator used before neural network integration.
"""

import math
import os
import sys
from typing import Dict, List, Tuple, Optional
import random
import numpy as np

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chaturanga import get_all_legal_moves, get_game_outcome, make_move

# Type alias for a move: ((start_row, start_col), (end_row, end_col))
Move = Tuple[Tuple[int, int], Tuple[int, int]]

class Node:
    """
    Represents a single state in the Monte Carlo Tree Search.
    Tracks PUCT metrics and parent/child tree relationships.
    """
    def __init__(self, current_player: int, prior_prob: float = 0.0, parent: 'Optional[Node]' = None, move: Optional[Move] = None):
        # Tree relationships
        self.parent = parent
        self.move = move  # The action from the parent that led to this state
        self.children: Dict[Move, 'Node'] = {}
        self.current_player = current_player
        
        # PUCT Algorithm Metrics
        self.N: int = 0          # Visit count
        self.W: float = 0.0      # Total accumulated value from backpropagation
        self.Q: float = 0.0      # Mean action value (W/N)
        self.P: float = prior_prob # Prior probability predicted by the policy network
        
    def is_expanded(self) -> bool:
        """Returns True if the node has been expanded (has children)."""
        return len(self.children) > 0

    def expand(self, action_probs: Dict[Move, float]) -> None:
        """
        Expands the node by creating child edges for all legal moves.
        action_probs: A dictionary mapping legal moves to their prior probability (P).
        """
        for move, prob in action_probs.items():
            if move not in self.children:
                self.children[move] = Node(
                    current_player=-self.current_player,
                    prior_prob=prob,
                    parent=self,
                    move=move
                )
                
    def get_puct_value(self, c_puct: float = 1.25) -> float:
        """
        Calculates the PUCT value used during the selection phase.
        Balances exploitation (Q) with exploration (P and visit counts).
        """
        parent_visits = self.parent.N if self.parent else 1
        
        # PUCT formula
        exploration_term = c_puct * self.P * (math.sqrt(parent_visits) / (1 + self.N))
        return self.Q + exploration_term

    def update(self, value: float) -> None:
        """
        Backpropagates the value from a leaf node evaluation.
        Updates the visit count and recalculates the mean action value.
        """
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

def dummy_evaluate(board: np.ndarray, legal_moves: List[Move]) -> Tuple[Dict[Move, float], float]:
    """
    A placeholder for the future PyTorch ResNet model.
    Mimics the Policy Head and Value Head outputs.
    """
    # 1. Dummy Value Head: Output a scalar between -1 (loss) and 1 (win)
    # In a real network, this evaluates how good the board state is for the current player.
    value = random.uniform(-1.0, 1.0)
    
    # 2. Dummy Policy Head: Output a probability distribution over the legal moves
    # In a real network, this will be a softmax output over a 4,096-dimensional action space
    # where illegal moves are masked out. Here, we just give them equal uniform probabilities.
    action_probs = {}
    if legal_moves:
        uniform_prob = 1.0 / len(legal_moves)
        for move in legal_moves:
            action_probs[move] = uniform_prob
            
    return action_probs, value

class MCTS:
    """
    Monte Carlo Tree Search algorithm specialized for AlphaZero.
    Uses neural network prior probabilities to guide tree expansion.
    """
    def __init__(self, c_puct: float = 1.25, num_simulations: int = 50):
        self.c_puct = c_puct
        self.num_simulations = num_simulations # Restricted to 30-50 for hardware constraints

    def search(self, initial_board: np.ndarray, current_player: int, root_node: Optional[Node] = None) -> Node:
        """
        Runs the MCTS simulation loop to expand the game tree and evaluate branches.
        """
        if root_node is None:
            root_node = Node(current_player=current_player)

        for _ in range(self.num_simulations):
            node = root_node
            board = initial_board.copy()
            search_path = [node]

            # 1. SELECTION: Traverse down the tree until an unexpanded leaf node is reached
            while node.is_expanded():
                # Pick the child edge that maximizes the PUCT formula
                best_move, best_child = max(
                    node.children.items(),
                    key=lambda item: item[1].get_puct_value(self.c_puct)
                )
                board = make_move(board, best_move[0], best_move[1])
                node = best_child
                search_path.append(node)

            # 2. EXPANSION & EVALUATION
            # Check if this leaf node is a terminal state (Checkmate, Bare King, Stalemate)
            outcome = get_game_outcome(board, node.current_player)
            
            if outcome is not None:
                # The game ended. Value is the absolute outcome for the current player
                value = outcome 
            else:
                # Not a terminal state. Get legal moves and evaluate with the neural network.
                legal_moves = get_all_legal_moves(board, node.current_player)
                
                # Replace this with the PyTorch model later!
                action_probs, value = dummy_evaluate(board, legal_moves)
                
                # Expand the tree with the valid moves and their predicted prior probabilities
                node.expand(action_probs)

            # 3. BACKPROPAGATION: Pass the predicted value back up the tree
            self.backpropagate(search_path, value, node.current_player)

        return root_node

    def backpropagate(self, search_path: List[Node], value: float, leaf_player: int) -> None:
        """
        Walks back up the search path to update visit counts (N) and mean values (Q).
        """
        for node in reversed(search_path):
            # Because Chaturanga is a zero-sum game, a winning position for one player 
            # is a losing position for the other. We flip the value if the node's turn 
            # does not match the leaf node's turn.
            node_value = value if node.current_player == leaf_player else -value
            node.update(node_value)

    def get_best_move(self, root_node: Node) -> Move:
        """
        Phase 4: Action Selection
        Chooses the final move to play on the real board based on the highest visit count (N).
        """
        if not root_node.children:
            raise ValueError("Root node has no children. Cannot select a move.")
            
        # In AlphaZero, the most heavily visited node is chosen as the best move,
        # rather than the highest Q value, as high visits indicate robust exploration.
        best_move, best_child = max(
            root_node.children.items(),
            key=lambda item: item[1].N
        )
        return best_move