import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Bernoulli
import numpy as np
from nexus_pytorch.evaluation.multimodal.mmvae.model.model_img_components import ImageEncoder, ImageDecoder
from nexus_pytorch.evaluation.multimodal.mmvae.model.model_trj_components import TrajectoryEncoder, TrajectoryDecoder
from nexus_pytorch.evaluation.multimodal.mmvae.model.model_snd_components import SoundEncoder, SoundDecoder
from nexus_pytorch.evaluation.multimodal.mmvae.model.model_sym_components import SymbolEncoder, SymbolDecoder
from nexus_pytorch.evaluation.multimodal.mmvae.model.model_nxs_components import LinearEncoder, NexusEncoder

class MMVAEModel(nn.Module):
    def __init__(self, nx_info, img_info, trj_info, snd_info, sym_info, snd_vae, use_cuda=False):

        super(MMVAEModel, self).__init__()

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
            if x_img is not None:
                mu, logvar = self.img_nx_encoder(img_z)
            elif x_trj is not None:
                mu, logvar = self.trj_nx_encoder(trj_z)
            elif x_snd is not None:
                mu, logvar = self.snd_nx_encoder(snd_z)
            else:
                mu, logvar = self.sym_nx_encoder(sym_z)

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


    def forward(self, x):

        # Variables to fill
        outs = []
        mod_mus = []
        mod_logvars = []
        mod_zs = []
        nx_mus = []
        nx_logvars = []
        nx_zs = []
        nx_mod_zs = [[None for _ in range(len(x))] for _ in range(len(x))]
        nx_snd_logvars = []

        # Encode Image Data
        if x[0] is not None:
            # Lower-level
            img_out, img_z, img_mu, img_logvar = self.img_mod_vae(x[0])
            outs.append(img_out)
            mod_mus.append(img_mu)
            mod_logvars.append(img_logvar)
            mod_zs.append(img_z)

            # Upper-level
            img_msg = img_z.clone().detach()
            img_nx_mu, img_nx_logvar = self.img_nx_encoder(img_msg)
            img_nx_z = self.reparametrize(img_nx_mu, img_nx_logvar)
            nx_mus.append(img_nx_mu)
            nx_logvars.append(img_nx_logvar)
            nx_zs.append(img_nx_z)

            # Decode lower level
            # reconstruct inputs from top level
            img_nx_img_z = self.nx_img_decoder(img_nx_z)
            img_nx_trj_z = self.nx_trj_decoder(img_nx_z)
            img_nx_snd_z = self.nx_snd_decoder(img_nx_z)
            img_nx_sym_z = self.nx_sym_decoder(img_nx_z)

            nx_mod_zs[0][0] = img_nx_img_z
            nx_mod_zs[0][1] = img_nx_trj_z
            nx_mod_zs[0][2] = img_nx_snd_z
            nx_mod_zs[0][3] = img_nx_sym_z

        # Encode Trajectory data
        if x[1] is not None:

            # Lower-level
            trj_out, trj_z, trj_mu, trj_logvar = self.trj_mod_vae(x[1])
            outs.append(trj_out)
            mod_mus.append(trj_mu)
            mod_logvars.append(trj_logvar)
            mod_zs.append(trj_z)

            # Upper-level
            trj_msg = trj_z.clone().detach()
            trj_nx_mu, trj_nx_logvar = self.trj_nx_encoder(trj_msg)
            trj_nx_z = self.reparametrize(trj_nx_mu, trj_nx_logvar)
            nx_mus.append(trj_nx_mu)
            nx_logvars.append(trj_nx_logvar)
            nx_zs.append(trj_nx_z)

            # Decode lower level
            # reconstruct inputs from top level
            trj_nx_img_z = self.nx_img_decoder(trj_nx_z)
            trj_nx_trj_z = self.nx_trj_decoder(trj_nx_z)
            trj_nx_snd_z = self.nx_snd_decoder(trj_nx_z)
            trj_nx_sym_z = self.nx_sym_decoder(trj_nx_z)

            nx_mod_zs[1][0] = trj_nx_img_z
            nx_mod_zs[1][1] = trj_nx_trj_z
            nx_mod_zs[1][2] = trj_nx_snd_z
            nx_mod_zs[1][3] = trj_nx_sym_z

        # Encode Sound data
        if x[2] is not None:
            # Lower-level
            with torch.no_grad():
                self.snd_mod_vae.eval()
                _, _, snd_z, _, _ = self.snd_mod_vae(x[2])

            outs.append(None)
            mod_mus.append(None)
            mod_logvars.append(None)
            mod_zs.append(snd_z)

            # Upper-level
            snd_msg = snd_z.clone().detach()
            snd_nx_mu, snd_nx_logvar = self.snd_nx_encoder(snd_msg)
            snd_nx_z = self.reparametrize(snd_nx_mu, snd_nx_logvar)
            nx_mus.append(snd_nx_mu)
            nx_logvars.append(snd_nx_logvar)
            nx_zs.append(snd_nx_z)

            # Decode lower level
            # reconstruct inputs from top level
            snd_nx_img_z = self.nx_img_decoder(snd_nx_z)
            snd_nx_trj_z = self.nx_trj_decoder(snd_nx_z)
            snd_nx_snd_z = self.nx_snd_decoder(snd_nx_z)
            snd_nx_sym_z = self.nx_sym_decoder(snd_nx_z)

            nx_mod_zs[2][0] = snd_nx_img_z
            nx_mod_zs[2][1] = snd_nx_trj_z
            nx_mod_zs[2][2] = snd_nx_snd_z
            nx_mod_zs[2][3] = snd_nx_sym_z

        # Encode Symbol data
        if x[3] is not None:

            # Lower-level
            sym_out, sym_z, sym_mu, sym_logvar = self.sym_mod_vae(x[3])
            outs.append(sym_out)
            mod_mus.append(sym_mu)
            mod_logvars.append(sym_logvar)
            mod_zs.append(sym_z)

            # Upper-level
            sym_msg = sym_z.clone().detach()
            sym_nx_mu, sym_nx_logvar = self.sym_nx_encoder(sym_msg)
            sym_nx_z = self.reparametrize(sym_nx_mu, sym_nx_logvar)
            nx_mus.append(sym_nx_mu)
            nx_logvars.append(sym_nx_logvar)
            nx_zs.append(sym_nx_z)

            # Decode lower level
            # reconstruct inputs from top level
            sym_nx_img_z = self.nx_img_decoder(sym_nx_z)
            sym_nx_trj_z = self.nx_trj_decoder(sym_nx_z)
            sym_nx_snd_z = self.nx_snd_decoder(sym_nx_z)
            sym_nx_sym_z = self.nx_sym_decoder(sym_nx_z)

            nx_mod_zs[3][0] = sym_nx_img_z
            nx_mod_zs[3][1] = sym_nx_trj_z
            nx_mod_zs[3][2] = sym_nx_snd_z
            nx_mod_zs[3][3] = sym_nx_sym_z

        # Compute optimal logvars for sigmaVAE training
        nx_snd_logvars.append(((mod_zs[2].clone().detach() - nx_mod_zs[0][2]) ** 2).mean([0, 1], keepdim=True).sqrt().log())
        nx_snd_logvars.append(
            ((mod_zs[2].clone().detach() - nx_mod_zs[1][2]) ** 2).mean([0, 1], keepdim=True).sqrt().log())
        nx_snd_logvars.append(
            ((mod_zs[2].clone().detach() - nx_mod_zs[2][2]) ** 2).mean([0, 1], keepdim=True).sqrt().log())
        nx_snd_logvars.append(
            ((mod_zs[2].clone().detach() - nx_mod_zs[3][2]) ** 2).mean([0, 1], keepdim=True).sqrt().log())

        return outs, mod_mus, mod_logvars, mod_zs, nx_mus, nx_logvars, nx_mod_zs, nx_snd_logvars



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
