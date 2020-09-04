import torch
import torch.optim as optim
import torch.nn.functional as F
from nexus_pytorch.evaluation.multimodal.mmvae.model.training_utils import AverageMeter, WarmUp, gaussian_nll


class Trainer(object):
    def __init__(self, model, training_config, cuda):

        # Model
        self.model = model
        self.use_cuda = cuda
        if cuda:
            self.model.cuda()

        # Training hyperparameters
        self.nx_drop_rate = training_config['nx_drop_rate']
        self.learning_rate = training_config['learning_rate']

        # Bottom-level training hyperparameters
        self.lambda_img = training_config['lambda_image']
        self.lambda_trj = training_config['lambda_trajectory']
        self.lambda_snd = training_config['lambda_sound']
        self.lambda_sym = training_config['lambda_symbol']
        self.beta_img = training_config['beta_image']
        self.beta_trj = training_config['beta_trajectory']
        self.beta_snd = training_config['beta_sound']
        self.beta_sym = training_config['beta_symbol']

        # Top-level training hyperparameters
        self.gamma_img = training_config['gamma_image']
        self.gamma_trj = training_config['gamma_trajectory']
        self.gamma_snd = training_config['gamma_sound']
        self.gamma_sym = training_config['gamma_symbol']
        self.beta_nx = training_config['beta_nexus']

        # Warmups
        self.wup_mod_epochs = training_config['wup_mod_epochs']
        self.wup_nx_epochs = training_config['wup_nx_epochs']
        self.beta_img_wup = WarmUp(epochs=self.wup_mod_epochs, value=self.beta_img)
        self.beta_trj_wup = WarmUp(epochs=self.wup_mod_epochs, value=self.beta_trj)
        self.beta_snd_wup = WarmUp(epochs=self.wup_mod_epochs, value=self.beta_snd)
        self.beta_sym_wup = WarmUp(epochs=self.wup_mod_epochs, value=self.beta_sym)
        self.betas_mod = [self.beta_img_wup, self.beta_trj_wup, self.beta_snd_wup, self.beta_sym_wup]
        self.beta_nx_wup = WarmUp(epochs=self.wup_nx_epochs, value=self.beta_nx)

        # Optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=training_config['learning_rate'])


    def loss(self, model, x):
        """Computes multi-modal vae """
        outs, mod_mus, mod_logvars, mod_zs, nx_mus, nx_logvars, nx_mod_zs, nx_snd_logvars = model(x)
        loss_mods = []

        # Targets
        img_z = mod_zs[0].clone().detach()
        trj_z = mod_zs[1].clone().detach()
        snd_z = mod_zs[2].clone().detach()
        sym_z = mod_zs[3].clone().detach()

        for d in range(len(outs)):

            # Mod prior term - For sound we do not compute this term
            if d != 2:
                mod_kld = self.betas_mod[d].get() * (
                        -0.5 * torch.sum(1 + mod_logvars[d] - mod_mus[d].pow(2) - mod_logvars[d].exp(), dim=1))
            else:
                mod_kld = 0.0

            # Reconstruction terms
            recons = []

            # For Image
            if d == 0:

                # Mod Recon term
                recons.append(self.lambda_img * torch.sum(
                    F.mse_loss(outs[d].view(outs[d].size(0), -1), x[d].view(x[d].size(0), -1), reduction='none'),
                    dim=-1))

            # For trajectory
            if d == 1:

                # Mod Recon term
                recons.append(self.lambda_trj * torch.sum(
                    F.mse_loss(outs[d].view(outs[d].size(0), -1), x[d].view(x[d].size(0), -1), reduction='none'),
                    dim=-1))

            # For sound:
            if d == 2:

                # No reconstruction
                continue

            # For symbol:
            if d == 3:
                _, targets = x[d].max(dim=1)
                recons.append(self.lambda_sym * F.cross_entropy(outs[d], targets, reduction='none'))


            ## Nexus Reconstruction terms
            for cm in range(len(nx_mod_zs[d])):

                # Image Nexus
                if cm == 0:

                    recons.append(
                        self.gamma_img * torch.sum(F.mse_loss(input=nx_mod_zs[d][cm], target=img_z, reduction='none'),
                                                 dim=-1))

                # Trajectory Nexus
                if cm == 1:
                    recons.append(
                        self.gamma_trj * torch.sum(F.mse_loss(input=nx_mod_zs[d][cm], target=trj_z, reduction='none'),
                            dim=-1))

                # Sound Nexus
                if cm == 2:
                    recons.append(
                        self.gamma_snd * \
                           torch.sum(gaussian_nll(nx_mod_zs[d][cm], nx_snd_logvars[d], snd_z), dim=-1))

                # Symbol Nexus
                if cm == 3:
                    recons.append(
                        self.gamma_sym * torch.sum(F.mse_loss(input=nx_mod_zs[d][cm], target=sym_z, reduction='none'),
                                                   dim=-1))

            # Nexus prior term terms
            nx_kld = self.beta_nx_wup.get() * (
                    -0.5 * torch.sum(1 + nx_logvars[d] - nx_mus[d].pow(2) - nx_logvars[d].exp(), dim=1))

            recon_mods = torch.stack(recons).sum(0)
            loss_mods.append(recon_mods + mod_kld + nx_kld)

        return torch.mean(torch.mean(torch.stack(loss_mods, dim=0), dim=0))




    def _run(self, train, epoch, dataloader, cuda):

        if train:
            str_name = 'Train'
            self.model.train()
        else:
            str_name = 'Eval'
            self.model.eval()

        # Meters
        loss_meter = AverageMeter()

        for batch_idx, modality_data in enumerate(dataloader):

            img_data = modality_data[1]
            trj_data = modality_data[2]
            snd_data = modality_data[3]
            sym_data = torch.nn.functional.one_hot(modality_data[0], num_classes=10).float()
            bs = img_data.size(0)

            if cuda:
                img_data = img_data.cuda()
                trj_data = trj_data.cuda()
                snd_data = snd_data.cuda()
                sym_data = sym_data.cuda()

            if train:
                self.optim.zero_grad()

            loss = self.loss(model=self.model, x=[img_data, trj_data, snd_data, sym_data])

            if train:
                loss.backward()
                self.optim.step()


            # Update meters
            loss_meter.update(loss.item(), bs)


            # log every 100 batches
            if batch_idx % 100 == 0:
                print(f'{str_name} Epoch: {epoch} '
                      f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                      f'Loss: {loss_meter.avg:.6f}')

        print(f'====> Epoch: {epoch}\t' f'Loss: {loss_meter.avg:.4f}')

        info = {
            'loss': loss_meter.avg
        }
        return info

    def train(self, epochs, dataset=None, cuda=True, post_epoch_cb=None, post_cb=None):
        # make them always callable, to avoid None checks later on
        if post_epoch_cb is None:
            post_epoch_cb = lambda x: None
        if post_cb is None:
            post_cb = lambda x: None

        if cuda:
            self.model.cuda()

        info = {}
        for epoch in range(1, epochs + 1):

            # Update Warmups
            self.beta_img_wup.update()
            self.beta_trj_wup.update()
            self.beta_snd_wup.update()
            self.beta_sym_wup.update()
            self.beta_nx_wup.update()

            is_training = True
            train_info = self._run(is_training, epoch, dataset.train_loader,
                                   cuda)
            is_training = False
            test_info = self._run(is_training, epoch, dataset.val_loader, cuda)

            info['model'] = self.model
            info['optimizer'] = self.optim
            info['epoch'] = epoch
            info['train'] = train_info
            info['test'] = test_info
            post_epoch_cb(info)

        post_cb(info)