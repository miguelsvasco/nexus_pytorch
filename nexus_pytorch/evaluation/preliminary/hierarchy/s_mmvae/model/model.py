from nexus_pytorch.evaluation.preliminary.hierarchy.s_mmvae.model.model_components import *

class MMVAE(nn.Module):
    def __init__(self, nx_info, img_info, sym_info, use_cuda=False):

        super(MMVAE, self).__init__()

        self.use_cuda = use_cuda
        self.img_info = img_info
        self.sym_info = sym_info
        self.nx_info = nx_info
        self.nx_dim = nx_info['nexus_dim']

        # Lower level representation
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

        # Process Data
        outs, _, _ = self.forward(x)

        return [outs[0][0],  outs[1][0]], [outs[1][1], outs[0][1]]


    def forward(self, x):

        # Variables to fill
        nx_mus = []
        nx_logvars = []
        nx_zs = []
        outs = [[None for _ in range(len(x))] for _ in range(len(x))]

        # Encode Image Data
        if x[0] is not None:

            # Encode
            img_h = self.img_mod_ae.encode(x[0])
            img_mu, img_logvar = self.img_nx_encoder(img_h)
            img_z = self.reparametrize(img_mu, img_logvar)

            # Append
            nx_mus.append(img_mu)
            nx_logvars.append(img_logvar)
            nx_zs.append(img_z)

            # Decode
            nx_img_z = self.nx_img_decoder(img_z)
            img_out = self.img_mod_ae.decode(nx_img_z)
            outs[0][0] = img_out

            cm_sym_z = self.nx_sym_decoder(img_z)
            cm_sym_out = self.sym_mod_ae.decode(cm_sym_z)
            outs[0][1] = cm_sym_out

        if x[1] is not None:

            # Encode
            sym_h = self.sym_mod_ae.encode(x[1])
            sym_mu, sym_logvar = self.sym_nx_encoder(sym_h)
            sym_z = self.reparametrize(sym_mu, sym_logvar)

            # Append
            nx_mus.append(sym_mu)
            nx_logvars.append(sym_logvar)
            nx_zs.append(sym_z)

            # Decode lower level
            cm_img_z = self.nx_img_decoder(sym_z)
            cm_img_out = self.img_mod_ae.decode(cm_img_z)
            outs[1][0] = cm_img_out

            nx_sym_z = self.nx_sym_decoder(sym_z)
            sym_out = self.sym_mod_ae.decode(nx_sym_z)
            outs[1][1] = sym_out

        return outs, nx_mus, nx_logvars


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