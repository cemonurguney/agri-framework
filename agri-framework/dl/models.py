import torch
import torch.nn as nn

def conv(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True)
    )

class UNetMini(nn.Module):
    def __init__(self, in_ch=3, base=16):
        super().__init__()
        self.c1 = conv(in_ch, base);   self.p1 = nn.MaxPool2d(2)
        self.c2 = conv(base, base*2);  self.p2 = nn.MaxPool2d(2)
        self.c3 = conv(base*2, base*4)
        self.u2 = nn.ConvTranspose2d(base*4, base*2, 2, 2)
        self.c4 = conv(base*4, base*2)
        self.u1 = nn.ConvTranspose2d(base*2, base, 2, 2)
        self.c5 = conv(base*2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        c1 = self.c1(x); x = self.p1(c1)
        c2 = self.c2(x); x = self.p2(c2)
        x  = self.c3(x)
        x  = self.u2(x); x = self.c4(torch.cat([x, c2], 1))
        x  = self.u1(x); x = self.c5(torch.cat([x, c1], 1))
        return self.out(x)
