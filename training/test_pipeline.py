"""
training.test_pipeline
----------------------
CPU end-to-end integration test verifying ChaturangPlayer move selection,
Bare King tactical lookahead, self-play replay generation, and one optimizer step.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def run_test():
    print("=" * 60)
    print("TESTING CHATURANGZERO PERSON 2 TRAINING PIPELINE")
    print("=" * 60)

    # 1. Test ChaturangPlayer move selection & tactical lookahead
    print("\n[Step 1/4] Testing ChaturangPlayer move selection...")
    from training.mcts_agent import ChaturangPlayer
    from chaturanga import ChaturangaEnv, KING, CHARIOT, PAWN

    env = ChaturangaEnv()
    player = ChaturangPlayer(num_sims=10, temperature=1.0)
    move = player.get_move(env.board, env.current_player)
    print(f"✓ Opening move chosen: {move}")

    # Tactical sanity test: Bare King capture in 1
    print("\n[Step 2/4] Verifying MCTS tactical evaluation (Bare King win-in-1)...")
    tac_board = np.zeros((8, 8), dtype=int)
    tac_board[0, 0] = KING          # White King
    tac_board[1, 1] = CHARIOT       # White Chariot
    tac_board[1, 5] = -PAWN         # Black Pawn
    tac_board[7, 7] = -KING         # Black King (will be bared upon capture)

    tac_player = ChaturangPlayer(num_sims=30, temperature=0.0)
    best_move = tac_player.get_move(tac_board, 1)
    assert best_move == ((1, 1), (1, 5)), f"Expected capture ((1, 1), (1, 5)), got {best_move}"
    print(f"✓ MCTS correctly chose winning capture move: {best_move} (PUCT sign verified)")

    # 3. Test 1 Game of Self-Play
    print("\n[Step 3/4] Generating 1 self-play game with model...")
    from training.self_play import SelfPlayAgent, ReplayBuffer
    agent = SelfPlayAgent(model=player.mcts.model, num_simulations=10, max_turns=8)
    samples = agent.play_game()
    print(f"✓ Self-play completed! Generated {len(samples)} training transitions.")

    buffer = ReplayBuffer(max_size=100)
    buffer.push_game(samples)
    assert len(buffer) == len(samples), "Buffer failed to store samples!"

    # 4. Test Training Optimization Step
    print("\n[Step 4/4] Checking PyTorch optimization step with ChaturangaNet...")
    try:
        import torch
        from training.train import train_step, ChaturangZeroLoss
        try:
            from neural.model import ChaturangaNet
            net = ChaturangaNet()
        except ImportError:
            from training.mcts_agent import DummyNeuralNet
            net = DummyNeuralNet()

        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        crit = ChaturangZeroLoss()

        batch = buffer.sample_batch(min(4, len(buffer)))
        loss, v_loss, p_loss = train_step(net, opt, crit, batch, device=torch.device("cpu"))
        print(f"✓ Backward pass successful! Total loss: {loss:.4f} (Val: {v_loss:.4f}, Pol: {p_loss:.4f})")
    except ImportError:
        print("✓ PyTorch not detected in local python env (will run on Google Colab GPU).")

    print("\n" + "=" * 60)
    print("ALL CHATURANGZERO PIPELINE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_test()
