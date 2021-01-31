import torch
import torch.optim as optim
import torch.nn.functional as F
from nexus_pytorch.evaluation.standard.mnist.mmvae.model.training_utils import AverageMeter, WarmUp


class Trainer(object):
    def __init__(self, model, training_config, cuda):

        # Model
        self.model = model
        self.use_cuda = cuda
        if cuda:
            self.model.cuda()

        # Training hyperparameters
        self.learning_rate = training_config['learning_rate']
        self.lambda_v = training_config['lambda_image']
        self.lambda_s = training_config['lambda_symbol']
        self.beta_nx = training_config['beta_nexus']

        # Warmups
        self.wup_nx_epochs = training_config['wup_nx_epochs']
        self.beta_nx_wup = WarmUp(epochs=self.wup_nx_epochs, value=self.beta_nx)

        # Optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=training_config['learning_rate'])

    def loss(self, model, x):
        """Computes multi-modal vae """
        outs, mus, logvars = model(x)
        loss_mods = []

        for d in range(len(outs)):

            # Divergence term
            kld =  self.beta_nx_wup.get() * (-0.5 * torch.sum(1 + logvars[d] - mus[d].pow(2) - logvars[d].exp(), dim=1))
            recons = []

            for cm in range(len(outs[d])):

                if cm == 0:  # MNIST
                    recons.append(self.lambda_v * torch.sum(
                        F.mse_loss(outs[d][cm].view(outs[d][cm].size(0), -1),
                                                           x[0].view(x[0].size(0), -1),
                                                           reduction='none'),
                        dim=-1))
                else:
                    _, targets = x[1].max(dim=1)
                    recons.append(self.lambda_s * F.cross_entropy(outs[d][cm], targets, reduction='none'))

            recon_mods = torch.stack(recons).sum(0)
            loss_mods.append(recon_mods + kld)

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

            m_data = modality_data[0]
            t_data = torch.nn.functional.one_hot(modality_data[1], num_classes=10).float()
            bs = m_data.size(0)

            if cuda:
                m_data = m_data.cuda()
                t_data = t_data.cuda()

            if train:
                self.optim.zero_grad()

            loss = self.loss(model=self.model, x=[m_data, t_data])

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
            'loss': loss_meter.avg,
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
