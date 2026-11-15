import torch
from torch import nn


def _block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SiameseUNet(nn.Module):
    """Shared-weight encoder on both dates; decoder sees |f1 - f2| at each scale.

    Small on purpose (~1M params): OSCD has only 14 training pairs.
    """

    def __init__(self, in_ch: int = 13, base: int = 16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.enc1, self.enc2 = _block(in_ch, c[0]), _block(c[0], c[1])
        self.enc3, self.enc4 = _block(c[1], c[2]), _block(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = _block(c[3] + c[2], c[2])
        self.dec2 = _block(c[2] + c[1], c[1])
        self.dec1 = _block(c[1] + c[0], c[0])
        self.head = nn.Conv2d(c[0], 1, 1)

    def _encode(self, x):
        f1 = self.enc1(x)
        f2 = self.enc2(self.pool(f1))
        f3 = self.enc3(self.pool(f2))
        f4 = self.enc4(self.pool(f3))
        return f1, f2, f3, f4

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        a, b = self._encode(before), self._encode(after)
        d1, d2, d3, d4 = (torch.abs(x - y) for x, y in zip(a, b))
        x = self.dec3(torch.cat([self.up(d4), d3], 1))
        x = self.dec2(torch.cat([self.up(x), d2], 1))
        x = self.dec1(torch.cat([self.up(x), d1], 1))
        return self.head(x)  # logits, shape (B, 1, H, W)
