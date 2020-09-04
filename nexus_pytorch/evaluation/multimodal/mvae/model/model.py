import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Bernoulli
import numpy as np
from nexus_pytorch.evaluation.multimodal.mvae.model.model_img_components import ImageEncoder, ImageDecoder
from nexus_pytorch.evaluation.multimodal.mvae.model.model_trj_components import TrajectoryEncoder, TrajectoryDecoder
from nexus_pytorch.evaluation.multimodal.mvae.model.model_snd_components import SoundEncoder, SoundDecoder
from nexus_pytorch.evaluation.multimodal.mvae.model.model_sym_components import SymbolEncoder, SymbolDecoder
from nexus_pytorch.evaluation.multimodal.mvae.model.model_nxs_components import LinearEncoder, NexusEncoder

class MVAEModel(nn.Module):
    def __init__(self, nx_info, img_info, trj_info, snd_info, sym_info, snd_vae, use_cuda=False):

        super(MVAEModel, self).__init__()

        # Parameters
        self.use_cuda = use_cuda
        self.nx_dim = nx_info['nexus_dim']
        self.img_info = img_info
        self.trj_info = trj_info
        self.snd_info = snd_info
        self.sym_info = sym_info
        self.aggregate_f = 'mean_dropout'
        self.nx_drop_rate = None

        # External Modality Representation
        # Image
        self.img_mod_vae = VAE(latent_dim=img_info['mod_latent_dim'], setup=self.img_info, mod='image', use_cuda=use_cuda)

        # Trajectory
        self.trj_mod_vae = VAE(latent_dim=trj_info['mod_latent_dim'], setup=self.trj_info, mod='trajectory',
                               use_cuda=use_cuda)
        # Sound - Pretrained representation
        self.snd_mod_vae = snd_vae

        # Internal Modality Representation
        # Symbol
        self.sym_mod_vae = VAE(latent_dim=sym_info['mod_latent_dim'], setup=self.sym_info, mod='symbol', use_cuda=use_cuda)
        
        # Nexus
        self.img_nx_encoder = LinearEncoder(name=str('nexus_img_encoder'),
                                          input_dim=img_info['mod_latent_dim'],
                                          layer_sizes=nx_info['layer_sizes'],
                                          output_dim=self.nx_dim)

        self.trj_nx_encoder = LinearEncoder(name=str('nexus_trj_encoder'),
                                            input_dim=trj_info['mod_latent_dim'],
                                            layer_sizes=nx_info['layer_sizes'],
                                            output_dim=self.nx_dim)

        self.snd_nx_encoder = LinearEncoder(name=str('nexus_snd_encoder'),
                                            input_dim=snd_info['mod_latent_dim'],
                                            layer_sizes=nx_info['layer_sizes'],
                                            output_dim=self.nx_dim)

        self.sym_nx_encoder = LinearEncoder(name=str('nexus_sym_encoder'),
                                          input_dim=sym_info['mod_latent_dim'],
                                          layer_sizes=nx_info['layer_sizes'],
                                          output_dim=self.nx_dim)


        self.nx_img_decoder = NexusEncoder(name=str('nexus_img_decoder'),
                                         input_dim=self.nx_dim,
                                         layer_sizes=np.flip(nx_info['layer_sizes']),
                                         output_dim=img_info['mod_latent_dim'])

        self.nx_trj_decoder = NexusEncoder(name=str('nexus_trj_decoder'),
                                           input_dim=self.nx_dim,
                                           layer_sizes=np.flip(nx_info['layer_sizes']),
                                           output_dim=trj_info['mod_latent_dim'])

        self.nx_snd_decoder = NexusEncoder(name=str('nexus_snd_decoder'),
                                           input_dim=self.nx_dim,
                                           layer_sizes=np.flip(nx_info['layer_sizes']),
                                           output_dim=snd_info['mod_latent_dim'])

        self.nx_sym_decoder = NexusEncoder(name=str('nexus_sym_decoder'),
                                         input_dim=self.nx_dim,
                                         layer_sizes=np.flip(nx_info['layer_sizes']),
                                         output_dim=sym_info['mod_latent_dim'])

        self.experts = ProductOfExperts()
        

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


    def generate(self, x_img=None, x_trj=None, x_snd=None, x_sym=None):

        with torch.no_grad():

            # Encode Image Modality Data
            if x_img is not None:
                img_out, img_z, _, _ = self.img_mod_vae(x_img)
            else:
                img_out, img_z = None, None

            # Encode Trajectory Modality Data
            if x_trj is not None:
                trj_out, trj_z, _, _ = self.trj_mod_vae(x_trj)
            else:
                trj_out, trj_z = None, None

            # Encode Trajectory Modality Data
            if x_snd is not None:
                snd_out, _, snd_z, _, _ = self.snd_mod_vae(x_snd)
            else:
                snd_out, snd_z = None, None

            # Encode Symbolic Modality Data
            if x_sym is not None:
                sym_out, sym_z, sym_mu, _ = self.sym_mod_vae(x_sym)
            else:
                sym_out, sym_z = None, None

            # Encode higher-level representation
            mu, logvar = self.infer(img_z, trj_z, snd_z, sym_z)

            # reparametrization trick to sample
            z = self.reparametrize(mu, logvar)

            # reconstruct inputs from top level
            img_z = self.nx_img_decoder(z)
            trj_z = self.nx_trj_decoder(z)
            snd_z = self.nx_snd_decoder(z)
            sym_z = self.nx_sym_decoder(z)

            nx_img_out = self.img_mod_vae.mod_decoder(img_z)
            nx_trj_out = self.trj_mod_vae.mod_decoder(trj_z)
            nx_snd_out = self.snd_mod_vae.mod_decoder(snd_z)
            nx_sym_out = self.sym_mod_vae.mod_decoder(sym_z)

            return [img_out, nx_img_out], [trj_out, nx_trj_out], [snd_out, nx_snd_out], [sym_out, nx_sym_out]

    def infer(self, img_msg=None, trj_msg=None, snd_msg=None, sym_msg=None):
        if img_msg is not None:
            batch_size = img_msg.size(0)
        elif trj_msg is not None:
            batch_size = trj_msg.size(0)
        elif snd_msg is not None:
            batch_size = snd_msg.size(0)
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

        if trj_msg is not None:
            trj_mu, trj_logvar = self.trj_nx_encoder(trj_msg)
            mu     = torch.cat((mu, trj_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, trj_logvar.unsqueeze(0)), dim=0)

        if snd_msg is not None:
            snd_mu, snd_logvar = self.snd_nx_encoder(snd_msg)
            mu     = torch.cat((mu, snd_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, snd_logvar.unsqueeze(0)), dim=0)

        if sym_msg is not None:
            sym_mu, sym_logvar = self.sym_nx_encoder(sym_msg)
            mu     = torch.cat((mu, sym_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, sym_logvar.unsqueeze(0)), dim=0)

        # product of experts to combine gaussians
        mu, logvar = self.experts(mu, logvar)
        return mu, logvar


    def forward(self, x_img=None, x_trj=None, x_snd=None, x_sym=None):

        # Encode Modality Data
        if x_img is not None:
            img_out, img_z, img_mu, img_logvar = self.img_mod_vae(x_img)
        else:
            img_out, img_z, img_mu, img_logvar = None, None, None, None

        if x_trj is not None:
            trj_out, trj_z, trj_mu, trj_logvar = self.trj_mod_vae(x_trj)
        else:
            trj_out, trj_z, trj_mu, trj_logvar = None, None, None, None

        if x_snd is not None:
            with torch.no_grad():
                self.snd_mod_vae.eval()
                snd_out, _, snd_z, _, _ = self.snd_mod_vae(x_snd)
        else:
            snd_out, snd_z = None, None

        if x_sym is not None:
            sym_out, sym_z, sym_mu, sym_logvar = self.sym_mod_vae(x_sym)
        else:
            sym_out, sym_z, sym_mu, sym_logvar = None, None, None, None

        # Encode higher-level representation
        if x_img is not None:
            img_msg = img_z.clone().detach()
        else:
            img_msg = None
        if x_trj is not None:
            trj_msg = trj_z.clone().detach()
        else:
            trj_msg = None
        if x_snd is not None:
            snd_msg = snd_z.clone().detach()
        else:
            snd_msg = None
        if x_sym is not None:
            sym_msg = sym_z.clone().detach()
        else:
            sym_msg = None
        nx_mu, nx_logvar = self.infer(img_msg, trj_msg, snd_msg, sym_msg)

        # reparametrization trick to sample
        z = self.reparametrize(nx_mu, nx_logvar)

        # reconstruct inputs
        nx_img_z = self.nx_img_decoder(z)
        nx_trj_z = self.nx_trj_decoder(z)
        nx_snd_z = self.nx_snd_decoder(z)
        nx_sym_z = self.nx_sym_decoder(z)

        # Compute log sigma optimal for sound
        if snd_z is not None:
            nx_snd_log_sigma = ((snd_z.clone().detach() - nx_snd_z) ** 2).mean([0, 1], keepdim=True).sqrt().log()
        else:
            nx_snd_log_sigma = None

        return img_out, trj_out, snd_out, sym_out, \
               [img_mu, img_logvar, img_z, nx_img_z], \
               [trj_mu, trj_logvar, trj_z, nx_trj_z], \
               [snd_z, nx_snd_z, nx_snd_log_sigma], \
               [sym_mu, sym_logvar, sym_z, nx_sym_z], nx_mu, nx_logvar



class VAE(nn.Module):
    def __init__(self, latent_dim, setup, mod, use_cuda=False):

        super(VAE, self).__init__()

        # Parameters
        self.latent_dim = latent_dim
        self.setup = setup
        self.mod = mod
        self.use_cuda = use_cuda

        # Components
        self.mod_encoder, self.mod_decoder = self.setup_model(setup)

    def setup_model(self, setup):

        if self.mod == 'image':
            mod_encoder = ImageEncoder(name=str('image mod_encoder'),
                                      input_dim=setup['input_dim'],
                                      n_channels=setup['input_channels'],
                                      conv_layers=setup['conv_layer_sizes'],
                                      linear_layers=setup['linear_layer_sizes'],
                                      output_dim=self.latent_dim)

            mod_decoder = ImageDecoder(name=str('image mod_decoder'),
                                      input_dim=self.latent_dim,
                                      n_channels=setup['input_channels'],
                                      conv_layers=np.flip(setup['conv_layer_sizes']),
                                      linear_layers=np.flip(setup['linear_layer_sizes']),
                                      output_dim=setup['input_dim'])

        elif self.mod == 'trajectory':
            mod_encoder = TrajectoryEncoder(name=str('trajectory mod_encoder'),
                                      input_dim=setup['input_dim'],
                                      layer_sizes=setup['linear_layer_sizes'],
                                      output_dim=self.latent_dim)

            mod_decoder = TrajectoryDecoder(name=str('trajectory mod_decoder'),
                                      input_dim=self.latent_dim,
                                      layer_sizes=np.flip(setup['linear_layer_sizes']),
                                      output_dim=setup['input_dim'])

        elif self.mod == 'sound':
            mod_encoder = SoundEncoder(output_dim=self.latent_dim)

            mod_decoder = SoundDecoder(input_dim=self.latent_dim)

        elif self.mod == 'symbol':
            mod_encoder = SymbolEncoder(name=str('symbol mod_encoder'),
                                      input_dim=setup['input_dim'],
                                      layer_sizes=setup['linear_layer_sizes'],
                                      output_dim=self.latent_dim)

            mod_decoder = SymbolDecoder(name=str('symbol mod_decoder'),
                                      input_dim=self.latent_dim,
                                      layer_sizes=np.flip(setup['linear_layer_sizes']),
                                      output_dim=setup['input_dim'])

        else:
            raise ValueError("Not implemented.")

        return mod_encoder, mod_decoder

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


    def forward(self, x, sample=True):

        mu, logvar = self.mod_encoder(x)
        z = self.reparametrize(mu, logvar)
        out = self.mod_decoder(z)
        return out, z, mu, logvar



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