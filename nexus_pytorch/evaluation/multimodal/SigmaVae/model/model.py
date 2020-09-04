import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Bernoulli
import numpy as np
from nexus_pytorch.evaluation.multimodal.SigmaVae.model.model_snd_components import SoundEncoder, SoundDecoder
import torch.nn.functional as F

def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor


class SigmaVAE(nn.Module):
    def __init__(self, latent_dim, use_cuda=False):

        super(SigmaVAE, self).__init__()

        # Parameters
        self.latent_dim = latent_dim
        self.use_cuda = use_cuda

        # Components
        self.mod_encoder = SoundEncoder(output_dim=self.latent_dim)
        self.mod_decoder = SoundDecoder(input_dim=self.latent_dim)


    def reparametrize(self, mu, logvar):

        # Sample epsilon from a random gaussian with 0 mean and 1 variance
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        # Check if cuda is selected
        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # std = exp(0.5 * log_var)
        std = logvar.mul(0.5).exp_()

        # z = std * epsilon + mu
        return mu.addcmul(std, epsilon)


    def forward(self, x):
        mu, logvar = self.mod_encoder(x)
        z = self.reparametrize(mu, logvar)
        out = self.mod_decoder(z)

        # Compute log_sigma optimal
        log_sigma = ((x - out) ** 2).mean([0, 1, 2, 3], keepdim=True).sqrt().log()

        # log_sigma = self.log_sigma
        out_log_sigma = softclip(log_sigma, -6)

        return out, out_log_sigma, z, mu, logvar