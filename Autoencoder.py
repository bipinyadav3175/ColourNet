import torch
import torch.nn as nn
import torch.nn.functional as F
from DataHandler import DataHandler

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        image_size = 512
        
        self.d1 = Down(1, 16)
        self.d2 = Down(16, 32)
        self.d3 = Down(32, 64)
        self.d4 = Down(64, 128)
        self.d5 = Down(128, 256)
        self.d6 = Down(256, 256)

        # The input dimensions of 'Up' blocks looks weird. Isn't it?
        # As we would be concatinating the outputs of 'Down' blocks with the inputs of 'Up' blocks(called skip connections), the input features increases
        self.u1 = Up(256, 256)
        self.u2 = Up(512, 128)
        self.u3 = Up(256, 64)
        self.u4 = Up(128, 32)
        self.u5 = Up(64, 16)
        self.u6 = Up(32, 3)

        self.conv_out = nn.Sequential(

            nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1)
        )

        self.tanh = nn.Tanh()


    def forward(self, x):
        noise = torch.randn_like(x) * 0.1

        dh = DataHandler()
        d1 = self.d1(x + noise)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        

        u1 = self.u1(d6)
        u2 = self.u2(u1, d5)
        u3 = self.u3(u2, d4)
        u4 = self.u4(u3, d3)
        u5 = self.u5(u4, d2)
        u6 = self.u6(u5, d1)


        out = self.conv_out(torch.cat((u6, x), dim=1))

        return self.tanh(out)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels) if activation else nn.Identity(),
            nn.ReLU() if activation else nn.Identity()
        )

    def forward(self, x1, x2=None):
        if x2 is None:
            x = x1
        else:
            x = torch.cat((x1, x2), dim=1)
        return self.layers(x)
    

if __name__ == "__main__":
    model = Autoencoder()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad)) # 2.35 M

