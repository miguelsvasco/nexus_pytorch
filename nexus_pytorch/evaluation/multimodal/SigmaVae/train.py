import os
import sys
import sacred
import random
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import nexus_pytorch.evaluation.multimodal.SigmaVae.ingredients as ingredients
from nexus_pytorch.evaluation.multimodal.SigmaVae.model.trainer import Trainer
from nexus_pytorch.evaluation.multimodal.SigmaVae.model.model import SigmaVAE
import nexus_pytorch.evaluation.multimodal.SigmaVae.model.training_utils as t_utils
from nexus_pytorch.scenarios.multimodal_dataset.multimodal_dataset import MultimodalDataset

ex = sacred.Experiment(
    'test_sound_sigma_vae',
    ingredients=[
        ingredients.training_ingredient, ingredients.model_ingredient,
        ingredients.model_debug_ingredient, ingredients.gpu_ingredient
    ])

@ex.capture
def log_dir_path(folder, _config):
    return os.path.join('.', f'results/log_{_config["seed"]}', folder)


@ex.capture
def exp_dir_path(folder, _config):
    return os.path.join('.', folder)


class PostEpochCb(object):
    def __init__(self, model, dataset):
        self.model = model
        self.sound_norm = dataset.get_sound_normalization()
        self.train_dataloader = dataset.train_loader
        self.val_dataloader = dataset.val_loader
        self.test_dataloader = dataset.get_test_loader(bsize=10)
        self.best_loss = sys.maxsize

    @ex.capture
    def _record_train_info(self, info, _run):
        _run.log_scalar('train_loss', info['loss'])
        _run.log_scalar('train_sound_recon_loss', info['snd_recon_loss'])
        _run.log_scalar('train_sound_prior_loss', info['snd_prior_loss'])

    @ex.capture
    def _record_test_info(self, info, _run):
        _run.log_scalar('test_loss', info['loss'])
        _run.log_scalar('test_sound_recon_loss', info['snd_recon_loss'])
        _run.log_scalar('test_sound_prior_loss', info['snd_prior_loss'])

    @ex.capture
    def _record_artifacts(self, info, _config):
        epoch = info['epoch']
        artifact_storage_interval = _config['model_debug'][
            'artifact_storage_interval']
        results_dir = log_dir_path('results')

        if epoch % artifact_storage_interval == 0:

            # Data
            with torch.no_grad():

                self.model.eval()
                test_data = next(iter(self.test_dataloader))
                snd_data = test_data[1]

                if self.model.use_cuda:
                    snd_data = snd_data.cuda()

                # Generate modalities from complete nexus
                snd_out, _,  _, _ , _  = self.model(snd_data)

                # Generate from prior
                s_z = torch.randn(snd_data.size(0), self.model.latent_dim)
                if self.model.use_cuda:
                    s_z = s_z.cuda()
                snd_prior_out = self.model.mod_decoder(s_z)


            # Trajectory
            if epoch == 0 or epoch == 1:
                t_utils.save_sound(data=snd_data, path=results_dir, filename='og_sound_e' + str(epoch),
                                   sound_norm=self.sound_norm)
                for i in range(3):
                    ex.add_artifact(os.path.join(results_dir, "og_sound_e" + str(epoch) + '_' + str(i) + '.wav'),
                                    name="og_sound_e" + str(epoch) + '_' + str(i) + ".wav")

            t_utils.save_sound(data=snd_out, path=results_dir, filename='sound_recon_e' + str(epoch),
                               sound_norm=self.sound_norm)
            for i in range(3):
                ex.add_artifact(os.path.join(results_dir, "sound_recon_e" + str(epoch) + '_' + str(i) + '.wav'),
                                name="sound_recon_e" + str(epoch) + '_' + str(i) + ".wav")

            t_utils.save_sound(data=snd_prior_out, path=results_dir, filename='sound_prior_e' + str(epoch),
                               sound_norm=self.sound_norm)
            for i in range(3):
                ex.add_artifact(os.path.join(results_dir, "sound_prior_e" + str(epoch) + '_' + str(i) + '.wav'),
                                name="sound_prior_e" + str(epoch) + '_' + str(i) + ".wav")



    @ex.capture
    def _record_checkpoint(self, info, _config):
        test_info = info['test']
        loss = test_info['loss']
        is_best = loss < self.best_loss
        self.best_loss = min(loss, self.best_loss)

        # Find a way to make this modality agnostic..... -- TODO
        model_config = dict(_config['model'])
        training_config = dict(_config['training'])

        t_utils.save_checkpoint(
            {
                'state_dict': self.model.state_dict(),
                'best_loss': self.best_loss,
                'loss': test_info['loss'],
                'epoch': info['epoch'],
                'optimizer': info['optimizer'].state_dict(),
                'model_config': model_config,
                'training_config': training_config
            },
            is_best,
            folder=log_dir_path('trained_models'))

    @ex.capture
    def __call__(self, epoch_info, _config):
        self._record_train_info(epoch_info['train'])
        self._record_test_info(epoch_info['test'])
        self._record_artifacts(epoch_info)
        self._record_checkpoint(epoch_info)


def post_cb(info):
    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'sound_vae_checkpoint.pth.tar'),
        name='sound_vae_last_checkpoint.pth.tar')
    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'best_sound_vae_model.pth.tar'),
        name='best_sound_vae_model.pth.tar')


@ex.capture
def train(_config, _run):

    # Read configs
    model_config = _config['model']
    training_config = _config['training']
    gpu_config = _config['gpu']

    # Set seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(training_config['seed'])
    np.random.seed(training_config['seed'])
    random.seed(training_config['seed'])
    torch.cuda.manual_seed(training_config['seed'])

    # Create Model
    sound_info = t_utils.get_specs(model_config)
    model = SigmaVAE(latent_dim=sound_info['mod_latent_dim'], use_cuda=gpu_config['cuda'])

    # Create trainer
    trainer = Trainer(model, training_config, gpu_config['cuda'])

    # Create Dataset
    dataset = MultimodalDataset(
        data_dir=exp_dir_path('../../../scenarios/multimodal_dataset/data'),
        modalities=["sound"],
        batch_size=training_config['batch_size'],
        eval_samples=training_config['eval_samples'],
        validation_size=training_config['validation_size'],
        seed=training_config['seed'])

    post_epoch_cb = PostEpochCb(model, dataset)

    trainer.train(epochs=training_config['epochs'], dataset=dataset, cuda=gpu_config['cuda'],
                  post_epoch_cb=post_epoch_cb, post_cb=post_cb)


@ex.main
def main(_config, _run):
    os.makedirs(log_dir_path('trained_models'), exist_ok=True)
    os.makedirs(log_dir_path('results'), exist_ok=True)

    train()


if __name__ == '__main__':
    ex.run_commandline()