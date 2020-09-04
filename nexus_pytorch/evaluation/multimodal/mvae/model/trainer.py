import torch
import torch.optim as optim
import torch.nn.functional as F
from nexus_pytorch.evaluation.multimodal.mvae.model.training_utils import AverageMeter, WarmUp, gaussian_nll


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
        self.beta_nx_wup = WarmUp(epochs=self.wup_nx_epochs, value=self.beta_nx)

        # Optimizer
        self.optim = optim.Adam(self.model.parameters(), lr=training_config['learning_rate'])

    def loss_subsampled(self, img_data=None, trj_data=None, snd_data=None, sym_data=None,
                        img_out=None, trj_out=None, sym_out=None,
                        img_dist=None, trj_dist=None, snd_dist=None, sym_dist=None,
                        nx_mu=None, nx_logvar=None):


        # Bottom representation losses
        img_vae_loss, trj_vae_loss, sym_vae_loss = 0.0, 0.0, 0.0

        if img_data is not None:
            img_vae_loss = self.gaussian_vae_loss_function(data=img_data, out=img_out,
                                                           mu=img_dist[0], logvar=img_dist[1],
                                                           lambda_value=self.lambda_img,
                                                           beta_value=self.beta_img_wup.get())

        if trj_data is not None:
            trj_vae_loss = self.gaussian_vae_loss_function(data=trj_data, out=trj_out,
                                                           mu=trj_dist[0], logvar=trj_dist[1],
                                                           lambda_value=self.lambda_trj,
                                                           beta_value=self.beta_trj_wup.get())

        if sym_data is not None:
            sym_vae_loss = self.bernoulli_vae_loss_function(data=sym_data, out=sym_out,
                                                           mu=sym_dist[0], logvar=sym_dist[1],
                                                           lambda_value=self.lambda_sym,
                                                           beta_value=self.beta_sym_wup.get())


        # Top representation loss
        img_nx_recon, trj_nx_recon, snd_nx_recon, sym_nx_recon = 0.0, 0.0, 0.0, 0.0

        # Nexus Loss
        if img_data is not None:
            img_z = img_dist[2].detach()
            # Image loss Terms
            img_nx_recon = self.gamma_img * torch.sum(F.mse_loss(input=img_dist[3],
                                                                 target=img_z,
                                                                 reduction='none'), dim=-1)
        if trj_data is not None:
            trj_z = trj_dist[2].detach()
            # Trajectory loss Terms
            trj_nx_recon = self.gamma_trj * torch.sum(F.mse_loss(input=trj_dist[3],
                                                                 target=trj_z,
                                                                 reduction='none'), dim=-1)
        if snd_data is not None:
            snd_z = snd_dist[0].detach()
            # Sound loss Terms
            snd_nx_recon = self.gamma_snd * \
                           torch.sum(gaussian_nll(snd_dist[1], snd_dist[2], snd_z), dim=-1)

        if sym_data is not None:
            sym_z = sym_dist[2].detach()
            # Symbol loss terms
            sym_nx_recon = self.gamma_sym * torch.sum(F.mse_loss(input=sym_dist[3],
                                                                 target=sym_z,
                                                                 reduction='none'), dim=-1)
        # Nexus
        nx_prior = self.beta_nx_wup.get() * \
                   (-0.5 * torch.sum(1 + nx_logvar - nx_mu.pow(2) - nx_logvar.exp(), dim=1))

        loss = torch.mean(img_vae_loss + trj_vae_loss + sym_vae_loss +
                          img_nx_recon + trj_nx_recon + snd_nx_recon + sym_nx_recon + nx_prior, dim=-1)

        return loss





    def gaussian_vae_loss_function(self, data, out, mu, logvar, lambda_value, beta_value):

        recon = lambda_value * torch.sum(F.mse_loss(out.view(out.size(0), -1),
                                                           data.view(data.size(0), -1),
                                                           reduction='none'), dim=-1)
        prior = beta_value * \
                        (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        vae_loss = recon + prior

        return vae_loss

    def bernoulli_vae_loss_function(self, data, out, mu, logvar, lambda_value, beta_value):

        _, targets = data.max(dim=1)
        recon = lambda_value * F.cross_entropy(out, targets, reduction='none')
        prior = beta_value * \
                        (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        vae_loss = recon + prior

        return vae_loss



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

            # Forward

            # Subsampling Forward
            # Complete modalities
            img_out, trj_out, snd_out, sym_out, \
            img_dist, trj_dist, snd_dist, sym_dist, nx_mu, nx_logvar = self.model(x_img=img_data,
                                                                                  x_trj=trj_data,
                                                                                  x_snd=snd_data,
                                                                                  x_sym=sym_data)

            loss_0 = self.loss_subsampled(img_data=img_data, trj_data=trj_data, sym_data=sym_data,
                                          img_out=img_out, trj_out=trj_out, sym_out=sym_out,
                                          img_dist=img_dist, trj_dist=trj_dist, snd_dist=snd_dist, sym_dist=sym_dist,
                                          nx_mu=nx_mu, nx_logvar=nx_logvar)


            # Single modalities
            img_img_out, img_trj_out, img_snd_out, img_sym_out, \
            img_img_dist, img_trj_dist, img_snd_dist, img_sym_dist, \
            img_nx_mu, img_nx_logvar = self.model(x_img=img_data,
                                                  x_trj=None,
                                                  x_snd=None,
                                                  x_sym=None)

            loss_1 = self.loss_subsampled(img_data=img_data, trj_data=None, sym_data=None,
                                          img_out=img_img_out, trj_out=None, sym_out=None,
                                          img_dist=img_img_dist, trj_dist=None, snd_dist=None, sym_dist=None,
                                          nx_mu=img_nx_mu, nx_logvar=img_nx_logvar)



            trj_img_out, trj_trj_out, trj_snd_out, trj_sym_out, \
            trj_img_dist, trj_trj_dist, trj_snd_dist, trj_sym_dist, \
            trj_nx_mu, trj_nx_logvar = self.model(x_img=None,
                                                  x_trj=trj_data,
                                                  x_snd=None,
                                                  x_sym=None)

            loss_2 = self.loss_subsampled(img_data=None, trj_data=trj_data,  sym_data=None,
                                          img_out=None, trj_out=trj_trj_out, sym_out=None,
                                          img_dist=None, trj_dist=trj_trj_dist, snd_dist=None, sym_dist=None,
                                          nx_mu=trj_nx_mu, nx_logvar=trj_nx_logvar)




            snd_img_out, snd_trj_out, snd_snd_out, snd_sym_out, \
            snd_img_dist, snd_trj_dist, snd_snd_dist, snd_sym_dist, \
            snd_nx_mu, snd_nx_logvar = self.model(x_img=None,
                                                  x_trj=None,
                                                  x_snd=snd_data,
                                                  x_sym=None)

            loss_3 = self.loss_subsampled(img_data=None, trj_data=None, snd_data=snd_data, sym_data=None,
                                          img_out=None, trj_out=None, sym_out=None,
                                          img_dist=None, trj_dist=None, snd_dist=snd_snd_dist, sym_dist=None,
                                          nx_mu=snd_nx_mu, nx_logvar=snd_nx_logvar)




            sym_img_out, sym_trj_out, sym_snd_out, sym_sym_out, \
            sym_img_dist, sym_trj_dist, sym_snd_dist, sym_sym_dist, \
            sym_nx_mu, sym_nx_logvar = self.model(x_img=None,
                                                  x_trj=None,
                                                  x_snd=None,
                                                  x_sym=sym_data)

            loss_4 = self.loss_subsampled(img_data=None, trj_data=None, sym_data=sym_data,
                                          img_out=None, trj_out=None, sym_out=sym_sym_out,
                                          img_dist=None, trj_dist=None, snd_dist=None, sym_dist=sym_sym_dist,
                                          nx_mu=sym_nx_mu, nx_logvar=sym_nx_logvar)



            # Two modalities
            imgtrj_img_out, imgtrj_trj_out, imgtrj_snd_out, imgtrj_sym_out, \
            imgtrj_img_dist, imgtrj_trj_dist, imgtrj_snd_dist, imgtrj_sym_dist, \
            imgtrj_nx_mu, imgtrj_nx_logvar = self.model(x_img=img_data,
                                                        x_trj=trj_data,
                                                        x_snd=None,
                                                        x_sym=None)

            loss_5 = self.loss_subsampled(img_data=img_data, trj_data=trj_data, sym_data=None,
                                          img_out=imgtrj_img_out, trj_out=imgtrj_trj_out, sym_out=None,
                                          img_dist=imgtrj_img_dist, trj_dist=imgtrj_trj_dist, snd_dist=None, sym_dist=None,
                                          nx_mu=imgtrj_nx_mu, nx_logvar=imgtrj_nx_logvar)




            imgsnd_img_out, imgsnd_trj_out, imgsnd_snd_out, imgsnd_sym_out, \
            imgsnd_img_dist, imgsnd_trj_dist, imgsnd_snd_dist, imgsnd_sym_dist, \
            imgsnd_nx_mu, imgsnd_nx_logvar = self.model(x_img=img_data,
                                                        x_trj=None,
                                                        x_snd=snd_data,
                                                        x_sym=None)

            loss_6 = self.loss_subsampled(img_data=img_data, trj_data=None, snd_data=snd_data, sym_data=None,
                                          img_out=imgsnd_img_out, trj_out=None, sym_out=None,
                                          img_dist=imgsnd_img_dist, trj_dist=None, snd_dist=imgsnd_snd_dist,
                                          sym_dist=None,
                                          nx_mu=imgsnd_nx_mu, nx_logvar=imgsnd_nx_logvar)



            imgsym_img_out, imgsym_trj_out, imgsym_snd_out, imgsym_sym_out, \
            imgsym_img_dist, imgsym_trj_dist, imgsym_snd_dist, imgsym_sym_dist, \
            imgsym_nx_mu, imgsym_nx_logvar = self.model(x_img=img_data,
                                                        x_trj=None,
                                                        x_snd=None,
                                                        x_sym=sym_data)

            loss_7 = self.loss_subsampled(img_data=img_data, trj_data=None, sym_data=sym_data,
                                          img_out=imgsym_img_out, trj_out=None, sym_out=imgsym_sym_out,
                                          img_dist=imgsym_img_dist, trj_dist=None, snd_dist=None, sym_dist=imgsym_sym_dist,
                                          nx_mu=imgsym_nx_mu, nx_logvar=imgsym_nx_logvar)




            trjsnd_img_out, trjsnd_trj_out, trjsnd_snd_out, trjsnd_sym_out, \
            trjsnd_img_dist, trjsnd_trj_dist, trjsnd_snd_dist, trjsnd_sym_dist, \
            trjsnd_nx_mu, trjsnd_nx_logvar = self.model(x_img=None,
                                                        x_trj=trj_data,
                                                        x_snd=snd_data,
                                                        x_sym=None)

            loss_8 = self.loss_subsampled(img_data=None, trj_data=trj_data, snd_data=snd_data, sym_data=None,
                                          img_out=None, trj_out=trjsnd_trj_out, sym_out=None,
                                          img_dist=None, trj_dist=trjsnd_trj_dist, snd_dist=trjsnd_snd_dist, sym_dist=None,
                                          nx_mu=trjsnd_nx_mu, nx_logvar=trjsnd_nx_logvar)



            trjsym_img_out, trjsym_trj_out, trjsym_snd_out, trjsym_sym_out, \
            trjsym_img_dist, trjsym_trj_dist, trjsym_snd_dist, trjsym_sym_dist, \
            trjsym_nx_mu, trjsym_nx_logvar = self.model(x_img=None,
                                                        x_trj=trj_data,
                                                        x_snd=None,
                                                        x_sym=sym_data)

            loss_9 = self.loss_subsampled(img_data=None, trj_data=trj_data, sym_data=sym_data,
                                          img_out=None, trj_out=trjsym_trj_out, sym_out=trjsym_sym_out,
                                          img_dist=None, trj_dist=trjsym_trj_dist, snd_dist=None, sym_dist=trjsym_sym_dist,
                                          nx_mu=trjsym_nx_mu, nx_logvar=trjsym_nx_logvar)



            sndsym_img_out, sndsym_trj_out, sndsym_snd_out, sndsym_sym_out, \
            sndsym_img_dist, sndsym_trj_dist, sndsym_snd_dist, sndsym_sym_dist, \
            sndsym_nx_mu, sndsym_nx_logvar = self.model(x_img=None,
                                                        x_trj=None,
                                                        x_snd=snd_data,
                                                        x_sym=sym_data)

            loss_10 = self.loss_subsampled(img_data=None, trj_data=None, sym_data=sym_data,
                                          img_out=None, trj_out=None, sym_out=sndsym_sym_out,
                                          img_dist=None, trj_dist=None, snd_dist=sndsym_snd_dist, sym_dist=sndsym_sym_dist,
                                          nx_mu=sndsym_nx_mu, nx_logvar=sndsym_nx_logvar)


            # Three modalities
            its_img_out, its_trj_out, its_snd_out, its_sym_out, \
            its_img_dist, its_trj_dist, its_snd_dist, its_sym_dist, \
            its_nx_mu, its_nx_logvar = self.model(x_img=img_data,
                                                        x_trj=trj_data,
                                                        x_snd=snd_data,
                                                        x_sym=None)

            loss_11 = self.loss_subsampled(img_data=img_data, trj_data=trj_data, snd_data=snd_data, sym_data=None,
                                          img_out=its_img_out, trj_out=its_trj_out, sym_out=None,
                                          img_dist=its_img_dist, trj_dist=its_trj_dist, snd_dist=its_snd_dist,
                                          sym_dist=None,
                                          nx_mu=its_nx_mu, nx_logvar=its_nx_logvar)


            tss_img_out, tss_trj_out, tss_snd_out, tss_sym_out, \
            tss_img_dist, tss_trj_dist, tss_snd_dist, tss_sym_dist, \
            tss_nx_mu, tss_nx_logvar = self.model(x_img=None,
                                                  x_trj=trj_data,
                                                  x_snd=snd_data,
                                                  x_sym=sym_data)

            loss_12 = self.loss_subsampled(img_data=None, trj_data=trj_data, snd_data=snd_data, sym_data=sym_data,
                                           img_out=None, trj_out=tss_trj_out, sym_out=tss_sym_out,
                                           img_dist=None, trj_dist=tss_trj_dist, snd_dist=tss_snd_dist, sym_dist=tss_sym_dist,
                                           nx_mu=tss_nx_mu, nx_logvar=tss_nx_logvar)



            ssi_img_out, ssi_trj_out, ssi_snd_out, ssi_sym_out, \
            ssi_img_dist, ssi_trj_dist, ssi_snd_dist, ssi_sym_dist, \
            ssi_nx_mu, ssi_nx_logvar = self.model(x_img=img_data,
                                                  x_trj=None,
                                                  x_snd=snd_data,
                                                  x_sym=sym_data)

            loss_13 = self.loss_subsampled(img_data=img_data, trj_data=None, snd_data=snd_data, sym_data=sym_data,
                                           img_out=ssi_img_out, trj_out=None, sym_out=ssi_sym_out,
                                           img_dist=ssi_img_dist, trj_dist=None, snd_dist=ssi_snd_dist, sym_dist=ssi_sym_dist,
                                           nx_mu=ssi_nx_mu, nx_logvar=ssi_nx_logvar)


            sit_img_out, sit_trj_out, sit_snd_out, sit_sym_out, \
            sit_img_dist, sit_trj_dist, sit_snd_dist, sit_sym_dist, \
            sit_nx_mu, sit_nx_logvar = self.model(x_img=img_data,
                                                  x_trj=trj_data,
                                                  x_snd=None,
                                                  x_sym=sym_data)

            loss_14 = self.loss_subsampled(img_data=img_data, trj_data=trj_data, sym_data=sym_data,
                                           img_out=sit_img_out, trj_out=sit_trj_out, sym_out=sit_sym_out,
                                           img_dist=sit_img_dist, trj_dist=sit_trj_dist, snd_dist=None, sym_dist=sit_sym_dist,
                                           nx_mu=sit_nx_mu, nx_logvar=sit_nx_logvar)


            total_loss = loss_0 + loss_1 + loss_2 \
                         + loss_3 + loss_4 + loss_5 \
                         + loss_6 + loss_7 + loss_8 \
                         + loss_9 + loss_10 + loss_11 + loss_12 + loss_13 + loss_14

            if train:
                total_loss.backward()
                self.optim.step()


            # Update meters
            loss_meter.update(total_loss.item(), bs)


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