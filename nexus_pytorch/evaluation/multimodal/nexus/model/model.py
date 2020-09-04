import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Bernoulli
import numpy as np
from nexus_pytorch.evaluation.multimodal.nexus.model.model_img_components import ImageEncoder, ImageDecoder
from nexus_pytorch.evaluation.multimodal.nexus.model.model_trj_components import TrajectoryEncoder, TrajectoryDecoder
from nexus_pytorch.evaluation.multimodal.nexus.model.model_snd_components import SoundEncoder, SoundDecoder
from nexus_pytorch.evaluation.multimodal.nexus.model.model_sym_components import SymbolEncoder, SymbolDecoder
from nexus_pytorch.evaluation.multimodal.nexus.model.model_nxs_components import LinearEncoder, NexusEncoder

class NexusModel(nn.Module):
    def __init__(self, nx_info, img_info, trj_info, snd_info, sym_info, snd_vae, use_cuda=False):

        super(NexusModel, self).__init__()

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
        nx_decoder_layers = nx_info['layer_sizes'] + [nx_info['message_dim']]
        self.img_msg_encoder = NexusEncoder(name=str('img_msg_encoder'),
                                          input_dim=img_info['mod_latent_dim'],
                                          layer_sizes=[],
                                          output_dim=nx_info['message_dim'])

        self.trj_msg_encoder = NexusEncoder(name=str('trj_msg_encoder'),
                                            input_dim=trj_info['mod_latent_dim'],
                                            layer_sizes=[],
                                            output_dim=nx_info['message_dim'])

        self.snd_msg_encoder = NexusEncoder(name=str('snd_msg_encoder'),
                                            input_dim=snd_info['mod_latent_dim'],
                                            layer_sizes=[],
                                            output_dim=nx_info['message_dim'])
        self.sym_msg_encoder = NexusEncoder(name=str('sym_msg_encoder'),
                                          input_dim=sym_info['mod_latent_dim'],
                                          layer_sizes=[],
                                          output_dim=nx_info['message_dim'])

        self.nx_node = LinearEncoder(name=str('nexus_updater'),
                                     input_dim=nx_info['message_dim'],
                                     layer_sizes=nx_info['layer_sizes'],
                                     output_dim=self.nx_dim)
        self.nx_img_decoder = NexusEncoder(name=str('nexus_img_decoder'),
                                         input_dim=self.nx_dim,
                                         layer_sizes=nx_decoder_layers,
                                         output_dim=img_info['mod_latent_dim'])

        self.nx_trj_decoder = NexusEncoder(name=str('nexus_trj_decoder'),
                                           input_dim=self.nx_dim,
                                           layer_sizes=nx_decoder_layers,
                                           output_dim=trj_info['mod_latent_dim'])

        self.nx_snd_decoder = NexusEncoder(name=str('nexus_snd_decoder'),
                                           input_dim=self.nx_dim,
                                           layer_sizes=nx_decoder_layers,
                                           output_dim=snd_info['mod_latent_dim'])

        self.nx_sym_decoder = NexusEncoder(name=str('nexus_sym_decoder'),
                                         input_dim=self.nx_dim,
                                         layer_sizes=nx_decoder_layers,
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

    def set_nx_drop(self, nx_drop=None):
        if nx_drop is None:
            self.nx_drop_rate = None
        else:
            self.nx_drop_rate = nx_drop
        return
    
    
    def aggregate(self, img_msg=None, trj_msg=None, snd_msg=None, sym_msg=None):

        # For single messages no need to aggregate
        if img_msg is None and trj_msg is None and snd_msg is None:
            return sym_msg
        if sym_msg is None and img_msg is None and trj_msg is None:
            return snd_msg
        if snd_msg is None and sym_msg is None and img_msg is None:
            return trj_msg
        if trj_msg is None and snd_msg is None and sym_msg is None:
            return img_msg

        # Concatenate existing messages
        if img_msg is not None:
            comp_msg = torch.zeros(size=img_msg.size())
        elif trj_msg is not None:
            comp_msg = torch.zeros(size=trj_msg.size())
        elif snd_msg is not None:
            comp_msg = torch.zeros(size=snd_msg.size())
        else:
            comp_msg = torch.zeros(size=sym_msg.size())

        comp_msg = comp_msg.unsqueeze(0)
        if self.use_cuda:
            comp_msg = comp_msg.cuda()

        if img_msg is not None:
            comp_msg = torch.cat([comp_msg, img_msg.unsqueeze(0)], dim=0)
        if trj_msg is not None:
            comp_msg = torch.cat([comp_msg, trj_msg.unsqueeze(0)], dim=0)
        if snd_msg is not None:
            comp_msg = torch.cat([comp_msg, snd_msg.unsqueeze(0)], dim=0)
        if sym_msg is not None:
            comp_msg = torch.cat([comp_msg, sym_msg.unsqueeze(0)], dim=0)
        comp_msg = comp_msg[1:]

        # Aggregate
        if self.aggregate_f == 'mean':
            comp_msg = torch.mean(comp_msg, dim=0)
        elif self.aggregate_f == 'mean_dropout':
            comp_msg = self.mean_drop(comp_msg, [img_msg, trj_msg, snd_msg, sym_msg])
        elif self.aggregate_f == 'sum':
            comp_msg = torch.sum(comp_msg, dim=0)
        else:
            raise ValueError("Not implemented")
        return comp_msg

    def mean_drop(self, mean_msg, mod_msgs):

        # Compute mean message
        mean_msg = torch.mean(mean_msg, dim=0)

        if not self.training:
            return mean_msg
        else:
            # For each entry in batch: (During training we have all modalities available)
            for i in range(mean_msg.size(0)):

                drop_mask = Bernoulli(torch.tensor([self.nx_drop_rate])).sample()

                # If there is no dropout, we continue
                if torch.sum(drop_mask).item() == 0:
                    continue

                # If there is dropout, we randomly select the number and type of modalities to drop
                else:
                    n_mods_to_drop = torch.randint(low=1, high=len(mod_msgs), size=(1,)).item()
                    mods_to_drop = np.random.choice(range(len(mod_msgs)), size=n_mods_to_drop, replace=False)

                    prune_msg = torch.zeros(mod_msgs[0].size(-1))
                    prune_msg = prune_msg.unsqueeze(0)
                    if self.use_cuda:
                        prune_msg = prune_msg.cuda()

                    for j in range(len(mod_msgs)):
                        if j in mods_to_drop:
                            continue
                        else:
                            prune_msg = torch.cat([prune_msg, mod_msgs[j][i].unsqueeze(0)], dim=0)
                    prune_msg = prune_msg[1:]
                    mean_msg[i] = torch.mean(prune_msg, dim=0)

            return mean_msg

    def generate(self, x_img=None, x_trj=None, x_snd=None, x_sym=None):
        with torch.no_grad():

            # Encode Image Modality Data
            if x_img is not None:
                img_out, img_z, _,  _ = self.img_mod_vae(x_img)
            else:
                img_out, img_z = None, None

            # Encode Trajectory Modality Data
            if x_trj is not None:
                trj_out, trj_z, _, _ = self.trj_mod_vae(x_trj)
            else:
                trj_out, trj_z = None, None

            # Encode Trajectory Modality Data
            if x_snd is not None:
                snd_out, _, snd_z,  _, _ = self.snd_mod_vae(x_snd)
            else:
                snd_out, snd_z = None, None

            # Encode Symbolic Modality Data
            if x_sym is not None:
                sym_out, sym_z, sym_mu, _ = self.sym_mod_vae(x_sym)
            else:
                sym_out, sym_z = None, None

            # Agreggate messages
            if img_z is not None:
                img_msg = self.img_msg_encoder(img_z)
            else:
                img_msg = None

            if trj_z is not None:
                trj_msg = self.trj_msg_encoder(trj_z)
            else:
                trj_msg = None

            if snd_z is not None:
                snd_msg = self.snd_msg_encoder(snd_z)
            else:
                snd_msg = None

            if sym_z is not None:
                sym_msg = self.sym_msg_encoder(sym_z)
            else:
                sym_msg = None
            nx_msg = self.aggregate(img_msg, trj_msg, snd_msg, sym_msg)

            # Update nexus node
            nx_mu, nx_logvar = self.nx_node(nx_msg)
            nx_z = self.reparametrize(nx_mu, nx_logvar)

            # Decode Nexus
            # Image
            img_q_mod_z = self.nx_img_decoder(nx_z)
            img_q_out = self.img_mod_vae.mod_decoder(img_q_mod_z)

            # Trajectory
            trj_q_mod_z = self.nx_trj_decoder(nx_z)
            trj_q_out = self.trj_mod_vae.mod_decoder(trj_q_mod_z)

            # Sound
            snd_q_mod_z = self.nx_snd_decoder(nx_z)
            snd_q_out = self.snd_mod_vae.mod_decoder(snd_q_mod_z)

            # Internal modalities
            sym_q_mod_z = self.nx_sym_decoder(nx_z)
            sym_q_out = self.sym_mod_vae.mod_decoder(sym_q_mod_z)

        return [img_out, img_q_out], [trj_out, trj_q_out], [snd_out, snd_q_out], [sym_out, sym_q_out]


    def forward(self, x_img, x_trj, x_snd, x_sym):

        # Encode Modality Data
        img_out, img_z, img_mu, img_logvar = self.img_mod_vae(x_img)
        trj_out, trj_z, trj_mu, trj_logvar = self.trj_mod_vae(x_trj)
        with torch.no_grad():
            self.snd_mod_vae.eval()
            snd_out, snd_out_logvar, snd_z, snd_mu, snd_logvar = self.snd_mod_vae(x_snd)
        sym_out, sym_z, sym_mu, sym_logvar = self.sym_mod_vae(x_sym)

        # Agreggate messages
        img_msg = self.img_msg_encoder(img_z.clone().detach())
        trj_msg = self.trj_msg_encoder(trj_z.clone().detach())
        snd_msg = self.snd_msg_encoder(snd_z.clone().detach())
        sym_msg = self.sym_msg_encoder(sym_z.clone().detach())
        nx_msg = self.aggregate(img_msg, trj_msg, snd_msg, sym_msg)

        # Update nexus node
        nx_mu, nx_logvar = self.nx_node(nx_msg)
        nx_z = self.reparametrize(nx_mu, nx_logvar)

        # Decode information from nexus
        nx_img_z = self.nx_img_decoder(nx_z)
        nx_trj_z = self.nx_trj_decoder(nx_z)
        nx_sym_z = self.nx_sym_decoder(nx_z)

        nx_snd_z = self.nx_snd_decoder(nx_z)
        # Compute log sigma optimal
        nx_snd_log_sigma = ((snd_z.clone().detach() - nx_snd_z) ** 2).mean([0, 1], keepdim=True).sqrt().log()

        return img_out, trj_out, snd_out, sym_out,\
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