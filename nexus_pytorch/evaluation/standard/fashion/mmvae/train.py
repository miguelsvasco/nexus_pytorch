import os
import sys
import sacred
import random
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
import nexus_pytorch.evaluation.standard.fashion.mmvae.ingredients as ingredients
from nexus_pytorch.evaluation.standard.fashion.mmvae.model.trainer import Trainer
from nexus_pytorch.evaluation.standard.fashion.mmvae.model.model import MMVAE
import nexus_pytorch.evaluation.standard.fashion.mmvae.model.training_utils as t_utils
from nexus_pytorch.scenarios.standard_dataset.standard_dataset import StandardDataset

ex = sacred.Experiment(
    'fashion_mmvae_train',
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
        self.train_dataloader = dataset.train_loader
        self.val_dataloader = dataset.val_loader
        self.test_dataloader = dataset.test_loader
        self.best_loss = sys.maxsize

    @ex.capture
    def _record_train_info(self, info, _run):
        _run.log_scalar('train_loss', info['loss'])

    @ex.capture
    def _record_test_info(self, info, _run):
        _run.log_scalar('test_loss', info['loss'])


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
                m_data = test_data[0]
                t_data = torch.nn.functional.one_hot(test_data[1], num_classes=10).float()

                if self.model.use_cuda:
                    m_data = m_data.cuda()
                    t_data = t_data.cuda()

                # Generate modalities
                img_mod_outs, sym_mod_outs = self.model.generate([m_data, t_data])
                m_out, nx_t_m_out = img_mod_outs[0], img_mod_outs[1]
                t_out, nx_m_t_out = sym_mod_outs[0], sym_mod_outs[1]

                # Mnist Recon
                m_out_comp = torch.cat([m_data.view(-1, 1, 28, 28).cpu(), m_out.view(-1, 1, 28, 28).cpu()])
                nx_t_m_comp = torch.cat([m_data.view(-1, 1, 28, 28).cpu(), nx_t_m_out.view(-1, 1, 28, 28).cpu()])

                # Text Recon
                t_res = np.argmax(torch.log_softmax(t_out, dim=-1).cpu().numpy(), axis=1).tolist()
                t_res_str = ''
                for i, item in enumerate(t_res):
                    t_res_str += str(item) + " "

                nx_m_t_res = np.argmax(torch.log_softmax(nx_m_t_out, dim=-1).cpu().numpy(), axis=1).tolist()
                nx_m_t_res_str = ''
                for i, item in enumerate(nx_m_t_res):
                    nx_m_t_res_str += str(item) + " "


            # Save data

            torchvision.utils.save_image(torchvision.utils.make_grid(m_out_comp,
                                                                     padding=5,
                                                                     pad_value=.5,
                                                                     nrow=m_data.size(0)),
                                         os.path.join(results_dir, 'm_comp_e' + str(epoch) + '.png'))
            ex.add_artifact(os.path.join(results_dir, "m_comp_e" + str(epoch) + '.png'),
                            name="image_recon_e" + str(epoch) + '.png')


            torchvision.utils.save_image(torchvision.utils.make_grid(nx_t_m_comp,
                                                                     padding=5,
                                                                     pad_value=.5,
                                                                     nrow=m_data.size(0)),
                                         os.path.join(results_dir, 'nx_t_m_comp_e' + str(epoch) + '.png'))
            ex.add_artifact(os.path.join(results_dir, "nx_t_m_comp_e" + str(epoch) + '.png'),
                            name="symbol_nexus_image_recon_e" + str(epoch) + '.png')


            with open(os.path.join(results_dir,'t_res_str_e' + str(epoch) + '.txt'), "w") as symbol_file:
                print(t_res_str, file=symbol_file)
            ex.add_artifact(os.path.join(results_dir, "t_res_str_e" + str(epoch) + '.txt'),
                            name= "symbol_recon_e" + str(epoch) + '.txt')

            with open(os.path.join(results_dir,'nx_m_t_res_str_e' + str(epoch) + '.txt'), "w") as symbol_file:
                print(nx_m_t_res_str, file=symbol_file)
            ex.add_artifact(os.path.join(results_dir, "nx_m_t_res_str_e" + str(epoch) + '.txt'),
                            name= "image_nexus_symbol_recon_e" + str(epoch) + '.txt')

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
            log_dir_path('trained_models'), 'mmvae_checkpoint.pth.tar'),
        name='mmvae_last_checkpoint.pth.tar')
    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'best_mmvae_model.pth.tar'),
        name='best_mmvae_model.pth.tar')


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
    nx_info, img_info, sym_info = t_utils.get_specs(model_config)
    model = MMVAE(nx_info=nx_info,
                 img_info=img_info,
                 sym_info=sym_info,
                 use_cuda=gpu_config['cuda'])

    # Create trainer
    trainer = Trainer(model, training_config, gpu_config['cuda'])

    # Create Dataset
    dataset = StandardDataset(
        dataset='fashion',
        data_dir='./data',
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