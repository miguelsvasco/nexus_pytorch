from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from nexus_pytorch.evaluation.standard.fashion.mvae.model.model_components import *



class MVAE(nn.Module):
    """Multimodal Variational Autoencoder.
    @param n_latents: integer
                      number of latent dimensions
    """
    def __init__(self, img_info, sym_info, nx_info, use_cuda=True):
        super(MVAE, self).__init__()

        self.use_cuda = use_cuda
        self.img_info = img_info
        self.sym_info = sym_info
        self.nx_info = nx_info

        self.img_mod_ae = AE(bottleneck_dim=img_info['mod_latent_dim'], setup=self.img_info, mod='image',
                             use_cuda=use_cuda)

        # Internal Modality Representation
        self.sym_mod_ae = AE(bottleneck_dim=sym_info['mod_latent_dim'], setup=self.sym_info, mod='symbol',
                             use_cuda=use_cuda)

        # Upper level representation
        self.img_nx_encoder = LinearEncoder(name=str('img_nx_encoder'),
                                          input_dim=img_info['mod_latent_dim'],
                                          layer_sizes=nx_info['layer_sizes'],
                                          output_dim=nx_info['nexus_dim'])

        self.sym_nx_encoder = LinearEncoder(name=str('sym_nx_encoder'),
                                          input_dim=sym_info['mod_latent_dim'],
                                          layer_sizes=nx_info['layer_sizes'],
                                          output_dim=nx_info['nexus_dim'])

        self.nx_img_decoder = NexusEncoder(name=str('img_nx_decoder'),
                                          input_dim=nx_info['nexus_dim'],
                                          layer_sizes=np.flip(nx_info['layer_sizes']),
                                          output_dim=img_info['mod_latent_dim'])

        self.nx_sym_decoder = NexusEncoder(name=str('sym_nx_decoder'),
                                          input_dim=nx_info['nexus_dim'],
                                          layer_sizes=np.flip(nx_info['layer_sizes']),
                                          output_dim=sym_info['mod_latent_dim'])

        self.experts       = ProductOfExperts()
        self.nx_dim     = nx_info['nexus_dim']

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

    def generate(self, x_v=None, x_s=None):
        with torch.no_grad():

            # Encode External Modality Data
            if x_v is not None:
                img_h = self.img_mod_ae.encode(x_v)
            else:
                img_h = None

            # Encode Symbolic Modality Data
            if x_s is not None:
                sym_h = self.sym_mod_ae.encode(x_s)
            else:
                sym_h = None

            # Encode higher-level representation
            mu, logvar = self.infer(img_h, sym_h)

            # reparametrization trick to sample
            z = self.reparametrize(mu, logvar)

            # reconstruct inputs from top level
            img_z = self.nx_img_decoder(z)
            sym_z = self.nx_sym_decoder(z)

            img_out = self.img_mod_ae.decode(img_z)
            sym_out = self.sym_mod_ae.decode(sym_z)

            return img_out, sym_out



    def forward(self, x_v=None, x_s=None):

        # Encode Modality Data
        if x_v is not None:
            img_h = self.img_mod_ae.encode(x_v)
        else:
            img_h = None

        if x_s is not None:
            sym_h = self.sym_mod_ae.encode(x_s)
        else:
            sym_h = None

        # Encode representation
        mu, logvar = self.infer(img_h, sym_h)

        # reparametrization trick to sample
        z = self.reparametrize(mu, logvar)

        # reconstruct inputs
        img_nx_z = self.nx_img_decoder(z)
        sym_nx_z = self.nx_sym_decoder(z)

        img_out = self.img_mod_ae.decode(img_nx_z)
        sym_out = self.sym_mod_ae.decode(sym_nx_z)

        return img_out, sym_out,  mu, logvar

    def infer(self, img_msg=None, sym_msg=None):
        if img_msg is not None:
            batch_size = img_msg.size(0)
        else:
            batch_size = sym_msg.size(0)

        use_cuda   = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert
        mu, logvar = prior_expert((1, batch_size, self.nx_dim),
                                  use_cuda=use_cuda)
        if img_msg is not None:
            img_mu, img_logvar = self.img_nx_encoder(img_msg)
            mu     = torch.cat((mu, img_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, img_logvar.unsqueeze(0)), dim=0)

        if sym_msg is not None:
            sym_mu, sym_logvar = self.sym_nx_encoder(sym_msg)
            mu     = torch.cat((mu, sym_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, sym_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar



class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.
    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, logvar, eps=1e-8):
        var       = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T         = 1. / (var + eps)
        pd_mu     = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var    = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * F.sigmoid(x)


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).
    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = Variable(torch.zeros(size))
    logvar = Variable(torch.zeros(size))
    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar


class AE(nn.Module):
    def __init__(self, bottleneck_dim, setup, mod, use_cuda=False):

        super(AE, self).__init__()

        # Parameters
        self.bottleneck_dim = bottleneck_dim
        self.setup = setup
        self.mod = mod
        self.use_cuda = use_cuda

        # Components
        self.mod_encoder, self.mod_decoder = self.setup_model(setup)

    def setup_model(self, setup):

        if self.mod == 'image':
            mod_encoder = ImageAEEncoder(name=str('image mod_encoder'),
                                      input_dim=setup['input_dim'],
                                      n_channels=setup['input_channels'],
                                      conv_layers=setup['conv_layer_sizes'],
                                      linear_layers=setup['linear_layer_sizes'],
                                      output_dim=self.bottleneck_dim)

            mod_decoder = ImageAEDecoder(name=str('image mod_decoder'),
                                      input_dim=self.bottleneck_dim,
                                      n_channels=setup['input_channels'],
                                      conv_layers=np.flip(setup['conv_layer_sizes']),
                                      linear_layers=np.flip(setup['linear_layer_sizes']),
                                      output_dim=setup['input_dim'])

        elif self.mod == 'symbol':
            mod_encoder = SymbolAEEncoder(name=str('image mod_encoder'),
                                      input_dim=setup['input_dim'],
                                      layer_sizes=setup['linear_layer_sizes'],
                                      output_dim=self.bottleneck_dim)

            mod_decoder = SymbolAEDecoder(name=str('image mod_decoder'),
                                      input_dim=self.bottleneck_dim,
                                      layer_sizes=np.flip(setup['linear_layer_sizes']),
                                      output_dim=setup['input_dim'])

        else:
            raise ValueError("Not implemented.")

        return mod_encoder, mod_decoder

    def encode(self, x):
        return self.mod_encoder(x)

    def decode(self, h):
        return self.mod_decoder(h)