from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sacred
import random
import numpy as np
import shutil
import sys
import nexus_pytorch.evaluation.autoencoders.image.ingredients as ingredients
import torchvision
from nexus_pytorch.scenarios.multimodal_dataset.multimodal_dataset import MultimodalDataset

ex = sacred.Experiment(
    'image_autoencoder',
    ingredients=[ingredients.gpu_ingredient, ingredients.training_ingredient,
                 ingredients.model_debug_ingredient
                 ])


@ex.capture
def log_dir_path(folder, _config):
    return os.path.join('.', f'results/log_{_config["seed"]}', folder)


@ex.capture
def exp_dir_path(folder, _config):
    return os.path.join('.', folder)


def save_checkpoint(state,
                    is_best,
                    folder='./'):

    filename = 'image_ae_checkpoint.pth.tar'
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(folder, filename),
            os.path.join(folder, 'best_image_ae_model.pth.tar'))


@ex.capture
def record_checkpoint(model, loss, best_loss, optimizer, epoch, is_best):

    save_checkpoint(
        {
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'loss': loss,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
        },
        is_best,
        folder=log_dir_path('trained_models'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Extra Components
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Image_AE(nn.Module):
    def __init__(self, b_dim):
        super(Image_AE, self).__init__()
        self.b_dim = b_dim
        self.encoder_conv = nn.Sequential(
                                     nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
                                     Swish(),
                                     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
                                     Swish()
                                     )
        self.encoder_linear = nn.Sequential(nn.Linear(64 * 7 * 7, b_dim))

        self.decoder_linear = nn.Sequential(nn.Linear(b_dim, 64 * 7 * 7))

        self.decoder_conv = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=False),
                                          Swish(),
                                          nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1,
                                                    bias=False),
                                          nn.Sigmoid()
                                          )

    def encode(self, x):
            x = self.encoder_conv(x)
            x = x.view(x.size(0), -1)
            return self.encoder_linear(x)

    def decode(self, h):
            h = self.decoder_linear(h)
            h = h.view(h.size(0), 64, 7, 7)
            return self.decoder_conv(h)

    def forward(self, x):
        b = self.encode(x)
        return self.decode(b)


def train_epoch(model, train_loader, optimizer, epoch, cuda):
    model.train()

    # Meters
    loss_meter = AverageMeter()

    for batch_idx, data in enumerate(train_loader):

        image = data[1]
        bs = image.size(0)

        if cuda:
            image = image.cuda()

        optimizer.zero_grad()
        output = model(image)
        loss = torch.mean(torch.sum(F.mse_loss(output.view(output.size(0), -1),
                                               image.view(image.size(0), -1),
                             reduction='none'), dim=-1))
        loss.backward()
        optimizer.step()

        # Update meters
        loss_meter.update(loss.item(), bs)

        # log every 100 batches
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss_meter.avg:.6f}')

    print(f'====> Epoch: {epoch}\t' f'Loss: {loss_meter.avg:.4f}')

    return loss_meter.avg


def test_epoch(model, test_loader, cuda):
    model.eval()

    # Meters
    loss_meter = AverageMeter()

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):

            image = data[1]
            bs = image.size(0)
            if cuda:
                image = image.cuda()

            # Forward
            output = model(image)
            loss = torch.mean(torch.sum(F.mse_loss(output.view(output.size(0), -1),
                                                   image.view(image.size(0), -1),
                                                   reduction='none'), dim=-1))

            # Update meters
            loss_meter.update(loss.item(), bs)

    print(f'====>Test Loss: {loss_meter.avg:.4f}')

    return loss_meter.avg


@ex.capture
def train(_config, _run):

    # Read configs
    training_config = _config['training']
    device = torch.device("cuda" if _config['gpu']['cuda'] else "cpu")
    results_dir = log_dir_path('results')
    artifact_storage_interval = _config['model_debug'][
        'artifact_storage_interval']

    # Set seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(training_config['seed'])
    np.random.seed(training_config['seed'])
    random.seed(training_config['seed'])
    torch.cuda.manual_seed(training_config['seed'])

    # Create Classifier
    model = Image_AE(b_dim=training_config['b_dim']).to(device)
    epochs = training_config['epochs']

    # Create Dataset
    dataset = MultimodalDataset(
        modalities=['image'],
        data_dir=exp_dir_path('../../../scenarios/multimodal_dataset/data'),
        batch_size=training_config['batch_size'],
        eval_samples=10,
        validation_size=0.1,
        seed=training_config['seed'])

    train_loader, val_loader = dataset.train_loader, dataset.val_loader
    test_loader = dataset.get_test_loader(bsize=20)

    # Training objects
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    best_loss = sys.maxsize

    for epoch in range(1, epochs + 1):

        train_loss = train_epoch(model, train_loader, optimizer, epoch, cuda=_config['gpu']['cuda'])
        val_loss = test_epoch(model, val_loader, cuda=_config['gpu']['cuda'])

        _run.log_scalar('train_loss', train_loss)
        _run.log_scalar('val_loss', val_loss)

        # Best Loss
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        record_checkpoint(model=model, loss=val_loss, best_loss=best_loss,
                          optimizer=optimizer, epoch=epoch, is_best=is_best)


        if epoch % artifact_storage_interval == 0:

            # Data
            with torch.no_grad():

                model.eval()
                data = next(iter(test_loader))[1].to(device)

                # Generate modalities
                image_out = model(data)
                
                # Vision Recon
                image_comp = torch.cat([data.view(-1, 1, 28, 28).cpu(), image_out.view(-1, 1, 28, 28).cpu()])

                torchvision.utils.save_image(torchvision.utils.make_grid(image_comp,
                                                                         padding=5,
                                                                         pad_value=.5,
                                                                         nrow=data.size(0)),
                                             os.path.join(results_dir, 'image_ae_mod_e' + str(epoch) + '.png'))
                ex.add_artifact(os.path.join(results_dir, "image_ae_mod_e" + str(epoch) + '.png'),
                                name="image_ae_recon_e" + str(epoch) + '.png')



    # Final Saving
    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'image_ae_checkpoint.pth.tar'),
        name='image_ae_last_checkpoint.pth.tar')
    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'best_image_ae_model.pth.tar'),
        name='best_image_ae_model.pth.tar')




@ex.main
def main(_config, _run):
    os.makedirs(log_dir_path('trained_models'), exist_ok=True)
    os.makedirs(log_dir_path('results'), exist_ok=True)
    train()


if __name__ == '__main__':
    ex.run_commandline()