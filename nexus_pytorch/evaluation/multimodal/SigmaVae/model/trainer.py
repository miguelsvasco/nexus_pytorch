import torch
import torch.optim as optim
import torch.nn.functional as F
from nexus_pytorch.evaluation.multimodal.nexus.model.training_utils import AverageMeter, WarmUp
import numpy as np

def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

class Trainer(object):
    def __init__(self, model, training_config, cuda):

        # Model
        self.model = model
        self.use_cuda = cuda
        if cuda:
            self.model.cuda()

        # Training hyperparameters
        self.learning_rate = training_config['learning_rate']

        # Bottom-level training hyperparameters
        self.lambda_snd = training_config['lambda_sound']
        self.beta_snd = training_config['beta_sound']

        # Warmups
        self.wup_mod_epochs = training_config['wup_mod_epochs']
        self.beta_snd_wup = WarmUp(epochs=self.wup_mod_epochs, value=self.beta_snd)

        # Optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=training_config['learning_rate'])


    def loss_function(self, snd_data, snd_out, snd_out_log_sigma, mu, logvar):

        # Sound loss Terms
        recon = self.lambda_snd * torch.sum(gaussian_nll(snd_out, snd_out_log_sigma, snd_data), dim=-1).sum(-1).sum(-1)
        # Nexus
        prior = self.beta_snd * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        # Total loss
        loss = torch.mean(recon + prior)

        return loss, recon, prior


    def _run(self, train, epoch, dataloader, cuda):

        if train:
            str_name = 'Train'
            self.model.train()
        else:
            str_name = 'Eval'
            self.model.eval()

        # Meters
        loss_meter = AverageMeter()

        snd_recon_meter = AverageMeter()
        snd_prior_meter = AverageMeter()

        for batch_idx, modality_data in enumerate(dataloader):

            snd_data = modality_data[1]
            bs = snd_data.size(0)

            if cuda:
                snd_data = snd_data.cuda()

            if train:
                self.optim.zero_grad()

            # Forward
            out, out_logsigma, _, mu, logvar = self.model(snd_data)

            # Losses
            loss, recon_loss, prior_loss = self.loss_function(snd_data, out, out_logsigma, mu, logvar)

            if train:
                loss.backward()
                self.optim.step()

            # Update meters
            loss_meter.update(loss.item(), bs)

            snd_recon_meter.update(torch.mean(recon_loss).item(), bs)
            snd_prior_meter.update(torch.mean(prior_loss).item(), bs)


            # log every 100 batches
            if batch_idx % 100 == 0:
                print(f'{str_name} Epoch: {epoch} '
                      f'({100. * batch_idx / len(dataloader):.0f}%)]\t'
                      f'Loss: {loss_meter.avg:.6f}')

        print(f'====> Epoch: {epoch}\t' f'Loss: {loss_meter.avg:.4f}')

        info = {
            'loss': loss_meter.avg,

            'snd_recon_loss': snd_recon_meter.avg,
            'snd_prior_loss': snd_prior_meter.avg,
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
            self.beta_snd_wup.update()

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