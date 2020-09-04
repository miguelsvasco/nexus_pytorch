import torch
import torch.nn as nn
import torch.nn.functional as F

FRAME_SIZE = 512
CONTEXT_FRAMES = 32
SPECTROGRAM_BINS = FRAME_SIZE//2 + 1


class SoundEncoder(nn.Module):
    def __init__(self, output_dim):
        super(SoundEncoder, self).__init__()

        # Properties
        self.conv_layer_0 = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        # Output layer of the network
        self.fc_mu = nn.Linear(2048, output_dim)
        self.fc_logvar = nn.Linear(2048, output_dim)


    def forward(self, x):
        x = self.conv_layer_0(x)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        h = x.view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)


class SoundDecoder(nn.Module):
    def __init__(self, input_dim):
        super(SoundDecoder, self).__init__()

        self.upsampler = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU()
        )

        self.hallucinate_0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.hallucinate_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.hallucinate_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
        )


    def forward(self, z):
        z = self.upsampler(z)
        z = z.view(-1, 256, 8, 1)
        z = self.hallucinate_0(z)
        z = self.hallucinate_1(z)
        out = self.hallucinate_2(z)
        return F.sigmoid(out)



class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)