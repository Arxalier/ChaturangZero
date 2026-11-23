import torch, torch.nn as nn, torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(c)
    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(y)))

class ChaturangaNet(nn.Module):
    def __init__(self, channels=64, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(13, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.p_conv = nn.Conv2d(channels, 32, 1); self.p_bn = nn.BatchNorm2d(32)
        self.p_fc = nn.Linear(32 * 64, 4096)
        self.v_conv = nn.Conv2d(channels, 1, 1); self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(64, 64); self.v_fc2 = nn.Linear(64, 1)

    def forward(self, x, mask=None):
        x = self.tower(self.stem(x))
        p = self.p_fc(F.relu(self.p_bn(self.p_conv(x))).flatten(1))
        if mask is not None:
            p = p.masked_fill(~mask, -1e9)
        log_p = F.log_softmax(p, dim=1)                       # (B, 4096)
        v = F.relu(self.v_bn(self.v_conv(x))).flatten(1)
        v = torch.tanh(self.v_fc2(F.relu(self.v_fc1(v))))     # (B, 1)
        return log_p, v