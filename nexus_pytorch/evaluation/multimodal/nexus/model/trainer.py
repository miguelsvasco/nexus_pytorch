import torch
import torch.optim as optim
import torch.nn.functional as F
from nexus_pytorch.evaluation.multimodal.nexus.model.training_utils import AverageMeter, WarmUp, gaussian_nll


class Trainer(object):
    def __init__(self, model, training_config, cuda):

        # Model
        self.model = model
        self.use_cuda = cuda
        if cuda:
            self.model.cuda()

        # Training hyperparameters
        self.nx_drop_rate = training_config['nx_drop_rate']
        self.model.set_nx_drop(self.nx_drop_rate)
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
        self.beta_nx_wup = WarmUp(epochs=self.wup_nx_epochs, value=self.beta_nx)

        # Optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=training_config['learning_rate'])


    def gaussian_vae_loss_function(self, data, out, mu, logvar, lambda_value, beta_value):

        recon = lambda_value * torch.sum(F.mse_loss(out.view(out.size(0), -1),
                                                           data.view(data.size(0), -1),
                                                           reduction='none'), dim=-1)
        prior = beta_value * \
                        (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        vae_loss = torch.mean(recon + prior)

        return vae_loss, recon, prior

    def bernoulli_vae_loss_function(self, data, out, mu, logvar, lambda_value, beta_value):

        _, targets = data.max(dim=1)
        recon = lambda_value * F.cross_entropy(out, targets, reduction='none')
        prior = beta_value * \
                        (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        vae_loss = torch.mean(recon + prior)

        return vae_loss, recon, prior


    def nexus_loss_function(self, img_z, trj_z, snd_z, sym_z,
                            img_nx_z, trj_nx_z, snd_nx_z, snd_nx_logvar, sym_nx_z,
                            nx_mu, nx_logvar):

        # Image loss Terms
        img_nx_recon = self.gamma_img * torch.sum(F.mse_loss(input=img_nx_z,
                                                        target=img_z,
                                                        reduction='none'), dim=-1)

        # Trajectory loss Terms
        trj_nx_recon = self.gamma_trj * torch.sum(F.mse_loss(input=trj_nx_z,
                                                             target=trj_z,
                                                             reduction='none'), dim=-1)

        # Sound loss Terms
        snd_nx_recon = self.gamma_snd * \
                       torch.sum(gaussian_nll(snd_nx_z, snd_nx_logvar, snd_z), dim=-1)

        # Symbol loss terms
        sym_nx_recon = self.gamma_sym * torch.sum(F.mse_loss(input=sym_nx_z,
                                                          target=sym_z,
                                                          reduction='none'), dim=-1)
        # Nexus
        nx_prior = self.beta_nx_wup.get() * \
                      (-0.5 * torch.sum(1 + nx_logvar - nx_mu.pow(2) - nx_logvar.exp(), dim=1))

        # Total loss
        loss = torch.mean(img_nx_recon + trj_nx_recon + snd_nx_recon + sym_nx_recon + nx_prior)

        return loss, img_nx_recon, trj_nx_recon, snd_nx_recon, sym_nx_recon, nx_prior


    def _run(self, train, epoch, dataloader, cuda):

        if train:
            str_name = 'Train'
            self.model.train()
        else:
            str_name = 'Eval'
            self.model.eval()

        # Meters
        loss_meter = AverageMeter()

        img_recon_meter = AverageMeter()
        trj_recon_meter = AverageMeter()
        sym_recon_meter = AverageMeter()
        img_mod_prior_meter = AverageMeter()
        trj_mod_prior_meter = AverageMeter()
        sym_mod_prior_meter = AverageMeter()

        img_nx_recon_meter = AverageMeter()
        trj_nx_recon_meter = AverageMeter()
        snd_nx_recon_meter = AverageMeter()
        sym_nx_recon_meter = AverageMeter()
        nexus_prior_meter = AverageMeter()

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

            # Forward
            img_out, trj_out, snd_out, sym_out, \
            img_dist, trj_dist, snd_dist, sym_dist, \
            nx_mu, nx_logvar = self.model(img_data, trj_data, snd_data, sym_data)

            # Losses
            # Vae Loss
            img_vae_loss, img_recon, img_prior = self.gaussian_vae_loss_function(img_data, img_out,
                                                                                 img_dist[0], img_dist[1],
                                                                                 self.lambda_img,
                                                                                 self.beta_img_wup.get())

            trj_vae_loss, trj_recon, trj_prior = self.gaussian_vae_loss_function(trj_data, trj_out,
                                                                                 trj_dist[0], trj_dist[1],
                                                                                 self.lambda_trj,
                                                                                 self.beta_trj_wup.get())

            sym_vae_loss, sym_recon, sym_prior = self.bernoulli_vae_loss_function(sym_data, sym_out,
                                                                                  sym_dist[0], sym_dist[1],
                                                                                  self.lambda_sym,
                                                                                  self.beta_sym_wup.get())

            # Nexus Loss
            img_z = img_dist[2].detach()
            trj_z = trj_dist[2].detach()
            snd_z = snd_dist[0].detach()
            sym_z = sym_dist[2].detach()

            nexus_loss, \
            nx_img_recon, nx_trj_recon, \
            nx_snd_recon, nx_sym_recon, nx_prior = self.nexus_loss_function(img_z, trj_z, snd_z, sym_z,
                                                                            img_dist[3], trj_dist[3], snd_dist[1],
                                                                            snd_dist[2], sym_dist[3],
                                                                            nx_mu, nx_logvar)

            total_loss = nexus_loss + img_vae_loss + trj_vae_loss + sym_vae_loss

            if train:
                total_loss.backward()
                self.optim.step()

            # Update meters
            loss_meter.update(total_loss.item(), bs)

            img_recon_meter.update(torch.mean(img_recon).item(), bs)
            img_mod_prior_meter.update(torch.mean(img_prior).item(), bs)
            img_nx_recon_meter.update(torch.mean(nx_img_recon).item(), bs)

            trj_recon_meter.update(torch.mean(trj_recon).item(), bs)
            trj_mod_prior_meter.update(torch.mean(trj_prior).item(), bs)
            trj_nx_recon_meter.update(torch.mean(nx_trj_recon).item(), bs)

            snd_nx_recon_meter.update(torch.mean(nx_snd_recon).item(), bs)

            sym_recon_meter.update(torch.mean(sym_recon).item(), bs)
            sym_mod_prior_meter.update(torch.mean(sym_prior).item(), bs)
            sym_nx_recon_meter.update(torch.mean(nx_sym_recon).item(), bs)

            nexus_prior_meter.update(torch.mean(nx_prior).item(), bs)


            # log every 100 batches
            if batch_idx % 100 == 0:
                print(f'{str_name} Epoch: {epoch} '
                      f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                      f'Loss: {loss_meter.avg:.6f}')

        print(f'====> Epoch: {epoch}\t' f'Loss: {loss_meter.avg:.4f}')

        info = {
            'loss': loss_meter.avg,

            'img_recon_loss': img_recon_meter.avg,
            'img_mod_prior_loss': img_mod_prior_meter.avg,
            'img_nx_recon_loss': img_nx_recon_meter.avg,

            'trj_recon_loss': trj_recon_meter.avg,
            'trj_mod_prior_loss': trj_mod_prior_meter.avg,
            'trj_nx_recon_loss': trj_nx_recon_meter.avg,

            'snd_nx_recon_loss': snd_nx_recon_meter.avg,

            'sym_recon_loss': sym_recon_meter.avg,
            'sym_mod_prior_loss': sym_mod_prior_meter.avg,
            'sym_nx_recon_loss': sym_nx_recon_meter.avg,

            'nexus_prior_loss': nexus_prior_meter.avg
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