from nexus_pytorch.evaluation.preliminary.hierarchy.h_mmvae.model.model_components import *

class MMVAE(nn.Module):
    def __init__(self, nx_info, img_info, sym_info, use_cuda=False):

        super(MMVAE, self).__init__()

        self.use_cuda = use_cuda
        self.img_info = img_info
        self.sym_info = sym_info
        self.nx_info = nx_info
        self.nx_dim = nx_info['nexus_dim']

        # Lower level representation
        self.img_mod_vae = VAE(latent_dim=img_info['mod_latent_dim'], setup=self.img_info, mod='image',
                               use_cuda=use_cuda)
        self.sym_mod_vae = VAE(latent_dim=sym_info['mod_latent_dim'], setup=self.sym_info, mod='symbol',
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


    def generate(self, x):

        out_nxs = [[None for _ in range(len(x))] for _ in range(len(x))]

        # Process Data
        outs, _, _, _, _, _, nx_mod_zs = self.forward(x)

        # Data from nexus
        for i in range(len(nx_mod_zs)):

            if x[i] is not None:
                m_z = nx_mod_zs[i][0]
                t_z = nx_mod_zs[i][1]

                m_out = self.img_mod_vae.mod_decoder(m_z)
                t_out = self.sym_mod_vae.mod_decoder(t_z)
                out_nxs[i][0], out_nxs[i][1] = m_out, t_out

        return outs, out_nxs


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

        # Encode Image Data
        if x[0] is not None:

            # Lower-level
            m_out, m_z, m_mu, m_logvar = self.img_mod_vae(x[0])
            outs.append(m_out)
            mod_mus.append(m_mu)
            mod_logvars.append(m_logvar)
            mod_zs.append(m_z)

            # Upper-level
            m_msg = m_z.clone().detach()
            m_nx_mu, m_nx_logvar = self.img_nx_encoder(m_msg)
            m_nx_z = self.reparametrize(m_nx_mu, m_nx_logvar)
            nx_mus.append(m_nx_mu)
            nx_logvars.append(m_nx_logvar)
            nx_zs.append(m_nx_z)

            # Decode lower level
            nx_m_z = self.nx_img_decoder(m_nx_z)
            nx_mod_zs[0][0] = nx_m_z
            cm_t_z = self.nx_sym_decoder(m_nx_z)
            nx_mod_zs[0][1] = cm_t_z

        if x[1] is not None:
            # Lower-level
            t_out, t_z, t_mu, t_logvar = self.sym_mod_vae(x[1])
            outs.append(t_out)
            mod_mus.append(t_mu)
            mod_logvars.append(t_logvar)
            mod_zs.append(t_z)

            # Upper-level
            t_msg = t_z.clone().detach()
            t_nx_mu, t_nx_logvar = self.sym_nx_encoder(t_msg)
            t_nx_z = self.reparametrize(t_nx_mu, t_nx_logvar)
            nx_mus.append(t_nx_mu)
            nx_logvars.append(t_nx_logvar)
            nx_zs.append(t_nx_z)

            # Decode lower level
            cm_m_z = self.nx_img_decoder(t_nx_z)
            nx_mod_zs[1][0] = cm_m_z
            nx_t_z = self.nx_sym_decoder(t_nx_z)
            nx_mod_zs[1][1] = nx_t_z

        return outs, mod_mus, mod_logvars, mod_zs, nx_mus, nx_logvars, nx_mod_zs


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