import torch
from chaturanga import create_starting_board
from neural.model import ChaturangaNet
from neural.encoder import *

def test():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = ChaturangaNet().to(dev)
    x = torch.randn(8, 13, 8, 8, device=dev)
    p, v = net(x)
    assert p.shape == (8, 4096) and v.shape == (8, 1)
    p.sum().add(v.sum()).backward()
    assert all(q.grad is not None for q in net.parameters())
    b = create_starting_board()
    for pl in (1, -1):
        m = get_action_mask(b, pl)
        assert m.sum() > 0
        for a in m.nonzero()[0]:
            assert move_to_action(action_to_move(a)) == a
    assert board_to_tensor(b, 1).shape == (13, 8, 8)
    assert board_to_tensor(b, 1)[12].all() and not board_to_tensor(b, -1)[12].any()
    mask = torch.from_numpy(get_action_mask(b, 1)).unsqueeze(0).to(dev)
    lp, _ = net(board_to_tensor(b, 1).unsqueeze(0).to(dev), mask)
    assert abs(lp.exp().sum().item() - 1) < 1e-4 and lp.exp()[~mask].sum() < 1e-6
    print("All neural tests passed on", dev)

if __name__ == "__main__":
    test()