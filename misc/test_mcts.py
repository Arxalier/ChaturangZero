import os
import sys

# Ensure repository root and misc are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

from chaturanga import ChaturangaEnv, make_move
from mcts import MCTS

def test_mcts_agent():
    # 1. Initialize the game environment
    env = ChaturangaEnv()
    
    # 2. Initialize the search tree logic (restricted to 50 simulations for hardware constraints)
    mcts = MCTS(num_simulations=50)
    
    # 3. Run the 4-step simulation loop to build the tree
    # This uses the dummy evaluator to look ahead
    root_node = mcts.search(initial_board=env.board, current_player=env.current_player)
    
    # 4. Phase 4: Action Selection
    best_move = mcts.get_best_move(root_node)
    
    print(f"MCTS chose move: {best_move}")
    
    # 5. Play the move on the real game board
    env.step(best_move)
    env.render()

if __name__ == "__main__":
    test_mcts_agent()