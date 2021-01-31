import torch
import torch.optim as optim
import torch.nn.functional as F
from nexus_pytorch.evaluation.standard.mnist.mvae.model.training_utils import AverageMeter, WarmUp


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


    def loss_function(self, img_data=None, sym_data=None, img_out=None, sym_out=None, nx_mu=None, nx_logvar=None):

        # Modality Recon Terms
        img_recon, sym_recon = 0, 0  # default params

        # Vision loss Terms
        if img_data is not None:
            img_recon = self.lambda_v * torch.sum(F.mse_loss(img_out.view(img_out.size(0), -1),
                                                           img_data.view(img_data.size(0), -1),
                                                           reduction='none'), dim=-1)

        # Symbol loss terms
        if sym_data is not None:
            _, targets = sym_data.max(dim=1)
            sym_recon = self.lambda_s * F.cross_entropy(sym_out, targets, reduction='none')


        # Nexus
        nx_prior = self.beta_nx_wup.get() * \
                      (-0.5 * torch.sum(1 + nx_logvar - nx_mu.pow(2) - nx_logvar.exp(), dim=1))

        # Total loss
        loss = torch.mean(img_recon + sym_recon + nx_prior)

        return loss, img_recon, sym_recon, nx_prior


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
        sym_recon_meter = AverageMeter()
        nexus_prior_meter = AverageMeter()

        for batch_idx, modality_data in enumerate(dataloader):

            img_data = modality_data[0]
            sym_data = torch.nn.functional.one_hot(modality_data[1], num_classes=10).float()
            bs = img_data.size(0)

            if cuda:
                img_data = img_data.cuda()
                sym_data = sym_data.cuda()

            if train:
                self.optim.zero_grad()


            # Subsampling Forward
            img_out, sym_out, nx_mu, nx_logvar = self.model(x_v=img_data, x_s=sym_data)
            m_m_out, m_t_out, m_mu, m_logvar = self.model(x_v=img_data, x_s=None)
            t_m_out, t_t_out, t_mu, t_logvar = self.model(x_v=None, x_s=sym_data)

            # Losses
            loss, img_recon, sym_recon, nx_prior = self.loss_function(img_data=img_data, sym_data=sym_data,
                                                                      img_out=img_out, sym_out=sym_out,
                                                                      nx_mu=nx_mu, nx_logvar=nx_logvar)

            m_loss, _, _, _ = self.loss_function(img_data=img_data, sym_data=None,
                                                 img_out=m_m_out, sym_out=None,
                                                 nx_mu=m_mu, nx_logvar=m_logvar)

            t_loss, _, _, _ = self.loss_function(img_data=None, sym_data=sym_data,
                                                 img_out=None, sym_out=t_t_out,
                                                 nx_mu=t_mu, nx_logvar=t_logvar)

            total_train = loss + m_loss + t_loss

            if train:
                total_train.backward()
                self.optim.step()


            # Update meters
            loss_meter.update(loss.item(), bs)
            img_recon_meter.update(torch.mean(img_recon).item(), bs)
            sym_recon_meter.update(torch.mean(sym_recon).item(), bs)
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
            'sym_recon_loss': sym_recon_meter.avg,
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