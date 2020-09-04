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
import nexus_pytorch.evaluation.autoencoders.sound.ingredients as ingredients
import torchvision
import librosa
import torchaudio
from nexus_pytorch.scenarios.multimodal_dataset.multimodal_dataset import MultimodalDataset
from nexus_pytorch.scenarios.multimodal_dataset.utils.sound_utils import *

ex = sacred.Experiment(
    'sound_autoencoder',
    ingredients=[ingredients.gpu_ingredient, ingredients.training_ingredient,
                 ingredients.model_debug_ingredient
                 ])


def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

@ex.capture
def log_dir_path(folder, _config):
    return os.path.join('.', f'results/log_{_config["seed"]}', folder)


@ex.capture
def exp_dir_path(folder, _config):
    return os.path.join('.', folder)


def save_checkpoint(state,
                    is_best,
                    folder='./'):


    filename = 'sound_ae_checkpoint.pth.tar'
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(folder, filename),
            os.path.join(folder, 'best_sound_ae_model.pth.tar'))


def save_sound(data, path, filename, sound_norm):

    for i in range(data.size(0)):
        # Unormalize data
        wave = data[i].cpu() * (sound_norm['max'] - sound_norm['min']) + sound_norm['min']

        # Permute channels and remove channel
        wave = wave.permute(0, 2, 1).squeeze(0)

        # DB to Power
        wave = librosa.db_to_power(wave)

        # Generate wave using Griffin-Lim algorithm
        sound_wav = librosa.feature.inverse.mel_to_audio(wave.squeeze(0).data.numpy(),
                                                         sr=16000,
                                                         n_iter=60)
        # Save data
        f_filename = filename + "_" + str(i) + ".wav"
        torchaudio.save(os.path.join(path, f_filename), torch.from_numpy(sound_wav) * np.iinfo(np.int16).max, 16000)

    return

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


def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor



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


class Sound_AE(nn.Module):
    def __init__(self, b_dim):
        super(Sound_AE, self).__init__()
        self.b_dim = b_dim

        self.encoder = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(1, 128), stride=(1, 1), padding=0, bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 1), stride=(2, 1),
                                               padding=(1, 0), bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4, 1), stride=(2, 1),
                                               padding=(1, 0), bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU())

        self.encoder_fc = nn.Linear(2048, b_dim)

        self.decoder_fc = nn.Sequential(nn.Linear(b_dim, 2048),
                                     nn.BatchNorm1d(2048),
                                     nn.ReLU())

        self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0), bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(4, 1),
                                                        stride=(2, 1), padding=(1, 0), bias=False),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(),
                                     nn.ConvTranspose2d(in_channels=128, out_channels=1, kernel_size=(1, 128),
                                                        stride=(1, 1), padding=0, bias=False),
                                     nn.Sigmoid())


    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        return self.encoder_fc(h)

    def decode(self, h):
        h = self.decoder_fc(h)
        h = h.view(-1, 256, 8, 1)
        return self.decoder(h)



    def forward(self, x):
        b = self.encode(x)
        out = self.decode(b)

        # Compute log_sigma optimal
        log_sigma = ((x - out) ** 2).mean([0, 1, 2, 3], keepdim=True).sqrt().log()

        # log_sigma = self.log_sigma
        out_log_sigma = softclip(log_sigma, -6)

        return out, out_log_sigma


def train_epoch(model, train_loader, optimizer, epoch, cuda):
    model.train()

    # Meters
    loss_meter = AverageMeter()

    for batch_idx, data in enumerate(train_loader):

        sound = data[1]
        bs = sound.size(0)

        if cuda:
            sound = sound.cuda()

        optimizer.zero_grad()
        output, out_logsigma = model(sound)
        loss = torch.mean(torch.sum(gaussian_nll(output, out_logsigma, sound), dim=-1).sum(-1).sum(-1))
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

            sound = data[1]
            bs = sound.size(0)
            if cuda:
                sound = sound.cuda()

            # Forward
            output, out_logsigma = model(sound)
            loss = torch.mean(torch.sum(gaussian_nll(output, out_logsigma, sound), dim=-1).sum(-1).sum(-1))

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
    model = Sound_AE(b_dim=training_config['b_dim']).to(device)
    epochs = training_config['epochs']

    # Create Dataset
    dataset = MultimodalDataset(
        modalities=['sound'],
        data_dir=exp_dir_path('../../../scenarios/multimodal_dataset/data'),
        batch_size=training_config['batch_size'],
        eval_samples=10,
        validation_size=0.1,
        seed=training_config['seed'])

    sound_norm = dataset.get_sound_normalization()

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
                out, _ = model(data)

                # Trajectory
                save_sound(data=data, path=results_dir, filename='og_sound_e' + str(epoch),
                                   sound_norm=sound_norm)
                for i in range(3):
                    ex.add_artifact(os.path.join(results_dir, "og_sound_e" + str(epoch) + '_' + str(i) + '.wav'),
                                    name="og_sound_e" + str(epoch) + '_' + str(i) + ".wav")

                save_sound(data=out, path=results_dir, filename='sound_recon_e' + str(epoch),
                           sound_norm=sound_norm)
                for i in range(3):
                    ex.add_artifact(os.path.join(results_dir, "sound_recon_e" + str(epoch) + '_' + str(i) + '.wav'),
                                    name="sound_recon_e" + str(epoch) + '_' + str(i) + ".wav")



    # Final Saving
    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'sound_ae_checkpoint.pth.tar'),
        name='sound_ae_last_checkpoint.pth.tar')
    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'best_sound_ae_model.pth.tar'),
        name='best_sound_ae_model.pth.tar')




@ex.main
def main(_config, _run):
    os.makedirs(log_dir_path('trained_models'), exist_ok=True)
    os.makedirs(log_dir_path('results'), exist_ok=True)
    train()


if __name__ == '__main__':
    ex.run_commandline()