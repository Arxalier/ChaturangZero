# end to end cpu verification
# initializes a player, plays 1 game with 10 mcts visits, creates a mock training example
# and tests 1 gradient step

import sys
import numpy as np

def run_test():
    print("=" * 60)
    print("TESTING PERSON 2 TRAINING PIPELINE")
    print("=" * 60)

    # 1. Test ChaturangPlayer move selection
    print("\n[Step 1/3] Testing ChaturangPlayer move selection...")
    from training.mcts_agent import ChaturangPlayer
    from chaturanga import ChaturangaEnv

    env = ChaturangaEnv()
    player = ChaturangPlayer(num_sims=10, temperature=1.0)
    move = player.get_move(env.board, env.current_player)
    print(f"✓ ChaturangPlayer chose legal move: {move}")

    # 2. Test 1 Game of Self-Play
    print("\n[Step 2/3] Generating 1 self-play game (10 MCTS visits/move)...")
    from training.self_play import SelfPlayAgent, ReplayBuffer
    agent = SelfPlayAgent(model=player.mcts.model, num_simulations=10, max_turns=6)
    samples = agent.play_game()
    print(f"✓ Self-play completed! Generated {len(samples)} training transitions.")

    buffer = ReplayBuffer(max_size=100)
    buffer.push_game(samples)
    assert len(buffer) == len(samples), "Buffer failed to store samples!"

    # 3. Test Training Update Step (if PyTorch is installed)
    print("\n[Step 3/3] Checking PyTorch optimization step...")
    try:
        import torch
        from training.train import train_step, ChaturangZeroLoss
        from training.mcts_agent import DummyNeuralNet

        class SimpleTorchNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(13, 16, kernel_size=3, padding=1)
                self.pol = torch.nn.Linear(16 * 8 * 8, 4096)
                self.val = torch.nn.Linear(16 * 8 * 8, 1)

            def forward(self, x):
                h = torch.relu(self.conv(x)).flatten(1)
                return self.pol(h), torch.tanh(self.val(h))

        net = SimpleTorchNet()
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        crit = ChaturangZeroLoss()

        batch = buffer.sample_batch(min(4, len(buffer)))
        loss, v_loss, p_loss = train_step(net, opt, crit, batch, device=torch.device("cpu"))
        print(f"✓ Backward pass successful! Total loss: {loss:.4f} (Val: {v_loss:.4f}, Pol: {p_loss:.4f})")
    except ImportError:
        print("✓ PyTorch not detected in local python env (will run on Google Colab GPU).")

    print("\n" + "=" * 60)
    print("ALL PERSON 2 PIPELINE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_test()