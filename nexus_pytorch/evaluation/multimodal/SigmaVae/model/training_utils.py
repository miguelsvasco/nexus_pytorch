import os
import shutil
import torch
import torchvision
import librosa
import soundfile as sf
import torchaudio
from nexus_pytorch.evaluation.multimodal.SigmaVae.model.model import SigmaVAE
from nexus_pytorch.scenarios.multimodal_dataset.utils.trajectory_utils import *

SAMPLE_RATE = 8000
TRAINING_RMS_MEAN = 0.067
FRAME_SIZE = 512
HOP_SIZE = 256
N_MELS = 128

def get_specs(model_config):

    # Sound
    snd_info = {
        'linear_layer_sizes': model_config['sound_linear_layers'],
        'mod_latent_dim': model_config['sound_mod_latent_dim'],
    }

    return snd_info


def save_checkpoint(state,
                    is_best,
                    folder='./',
                    filename='sound_vae_checkpoint.pth.tar'):
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(folder, filename),
            os.path.join(folder, 'best_sound_vae_model.pth.tar'))


def load_checkpoint(checkpoint_file, use_cuda=False):
    if use_cuda:
        checkpoint = torch.load(checkpoint_file)
    else:
        checkpoint = torch.load(
            checkpoint_file, map_location=lambda storage, location: storage)

    model_config = checkpoint['model_config']
    snd_info = get_specs(model_config)
    model = SigmaVAE(latent_dim=snd_info['mod_latent_dim'], use_cuda=use_cuda)
    model.load_state_dict(checkpoint['state_dict'])

    if use_cuda:
        model.cuda()

    return model, checkpoint['training_config']



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


class WarmUp(object):
    "Returns the value of the anneling factor to be used"

    def __init__(self, epochs=100, value=1.0):
        self.epoch = 0
        self.max_epoch = epochs
        self.value = value

    def get(self):

        if self.epoch >= self.max_epoch:
            return self.value
        else:
            return self.value*(float(self.epoch)/self.max_epoch)

    def update(self):
        self.epoch += 1


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