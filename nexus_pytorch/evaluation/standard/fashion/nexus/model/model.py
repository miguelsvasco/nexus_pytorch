from nexus_pytorch.evaluation.standard.mnist.nexus.model.model_components import *


class NexusModel(nn.Module):
    def __init__(self, nx_info, img_info, sym_info, use_cuda=False):

        super(NexusModel, self).__init__()

        # Parameters
        self.use_cuda = use_cuda
        self.nx_dim = nx_info['nexus_dim']
        self.img_info = img_info
        self.sym_info = sym_info
        self.aggregate_f = 'mean_dropout'
        self.nx_drop_rate = None

        # External Modality Representation
        # Image
        self.img_mod_vae = VAE(latent_dim=img_info['mod_latent_dim'], setup=self.img_info, mod='image', use_cuda=use_cuda)


        # Internal Modality Representation
        self.sym_mod_vae = VAE(latent_dim=sym_info['mod_latent_dim'], setup=self.sym_info, mod='symbol', use_cuda=use_cuda)
        
        # Nexus
        nx_decoder_layers = nx_info['layer_sizes'] + [nx_info['message_dim']]
        self.img_msg_encoder = NexusEncoder(name=str('img_msg_encoder'),
                                          input_dim=img_info['mod_latent_dim'],
                                          layer_sizes=[],
                                          output_dim=nx_info['message_dim'])
        self.sym_msg_encoder = NexusEncoder(name=str('syimg_msg_encoder'),
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
    
    
    def aggregate(self, img_msg=None, sym_msg=None):

        # For single messages no need to aggregate
        if img_msg is None:
            return sym_msg
        if sym_msg is None:
            return img_msg

        # Concatenate existing messages
        comp_msg = torch.zeros(size=img_msg.size())
        comp_msg = comp_msg.unsqueeze(0)
        if self.use_cuda:
            comp_msg = comp_msg.cuda()
        comp_msg = torch.cat([comp_msg, img_msg.unsqueeze(0)], dim=0)
        comp_msg = torch.cat([comp_msg, sym_msg.unsqueeze(0)], dim=0)
        comp_msg = comp_msg[1:]

        # Aggregate
        if self.aggregate_f == 'mean':
            comp_msg = torch.mean(comp_msg, dim=0)
        elif self.aggregate_f == 'mean_dropout':
            comp_msg = self.mean_drop(comp_msg, [img_msg, sym_msg])
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

    def generate(self, x_v=None, x_s=None):
        with torch.no_grad():

            # Encode External Modality Data
            if x_v is not None:
                img_out, _, img_z,  _ = self.img_mod_vae(x_v)
            else:
                img_out, img_z = None, None

            # Encode Symbolic Modality Data
            if x_s is not None:
                sym_out, _, sym_z, _ = self.sym_mod_vae(x_s)
            else:
                sym_out, sym_z = None, None

            # Agreggate messages
            if img_z is not None:
                img_msg = self.img_msg_encoder(img_z)
            else:
                img_msg = None
            if sym_z is not None:
                sym_msg = self.sym_msg_encoder(sym_z)
            else:
                sym_msg = None
            nx_msg = self.aggregate(img_msg, sym_msg)

            # Update nexus node
            nx_mu, nx_logvar = self.nx_node(nx_msg)
            nx_z = self.reparametrize(nx_mu, nx_logvar)

            # Decode Qualia
            # External modalities
            img_q_mod_z = self.nx_img_decoder(nx_z)
            img_q_out = self.img_mod_vae.mod_decoder(img_q_mod_z)

            # Internal modalities
            sym_q_mod_z = self.nx_sym_decoder(nx_z)
            sym_q_out = self.sym_mod_vae.mod_decoder(sym_q_mod_z)

        return [img_out, img_q_out], [sym_out, sym_q_out]


    def forward(self, x_v, x_s):

        # Encode Modality Data
        img_out, img_z, img_mu, img_logvar = self.img_mod_vae(x_v)
        sym_out, sym_z, sym_mu, sym_logvar = self.sym_mod_vae(x_s)

        # Agreggate messages
        img_msg = self.img_msg_encoder(img_z.clone().detach())
        sym_msg = self.sym_msg_encoder(sym_z.clone().detach())
        nx_msg = self.aggregate(img_msg, sym_msg)

        # Update nexus node
        nx_mu, nx_logvar = self.nx_node(nx_msg)
        nx_z = self.reparametrize(nx_mu, nx_logvar)

        # Decode information from nexus
        nx_m_z = self.nx_img_decoder(nx_z)
        nx_t_z = self.nx_sym_decoder(nx_z)


        return img_out, sym_out,\
               [img_mu, img_logvar, img_z, nx_m_z],\
               [sym_mu, sym_logvar, sym_z, nx_t_z], nx_mu, nx_logvar



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

        elif self.mod == 'symbol':
            mod_encoder = SymbolEncoder(name=str('image mod_encoder'),
                                      input_dim=setup['input_dim'],
                                      layer_sizes=setup['linear_layer_sizes'],
                                      output_dim=self.latent_dim)

            mod_decoder = SymbolDecoder(name=str('image mod_decoder'),
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