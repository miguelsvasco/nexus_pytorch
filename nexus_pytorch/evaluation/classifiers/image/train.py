from __future__ import print_function
import os
import torch
import torch.nn as nn
import sacred
import random
import numpy as np
import shutil
import sys
import nexus_pytorch.evaluation.classifiers.image.ingredients as ingredients
from nexus_pytorch.scenarios.multimodal_dataset.multimodal_dataset import MultimodalDataset

ex = sacred.Experiment(
    'image_classifier_train',
    ingredients=[ingredients.gpu_ingredient, ingredients.training_ingredient
                 ])


@ex.capture
def log_dir_path(folder, _config):
    return os.path.join('.', f'results/log_{_config["seed"]}', folder)


@ex.capture
def exp_dir_path(folder, _config):
    return os.path.join('.', folder)


def save_checkpoint(state,
                    is_best,
                    folder='./',
                    filename='image_classifier_checkpoint.pth.tar'):
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(folder, filename),
            os.path.join(folder, 'best_image_classifier_model.pth.tar'))


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


class Image_Classifier(nn.Module):
    def __init__(self):
        super(Image_Classifier, self).__init__()
        self.cnn_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.cnn_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.dropout2d = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        out = self.cnn_1(x)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)

        out = self.cnn_2(out)
        out = self.relu(out)
        out = self.dropout2d(out)
        out = self.maxpool(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.out(out)

        return out


def train_epoch(model, train_loader, optimizer, criterion, epoch, cuda):
    model.train()

    # Meters
    loss_meter = AverageMeter()

    for batch_idx, modality_data in enumerate(train_loader):

        m_data = modality_data[1]
        labels = modality_data[0]
        bs = m_data.size(0)

        if cuda:
            m_data = m_data.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()

        output = model(m_data)
        loss = criterion(output, labels)
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


def test_epoch(model, test_loader, criterion, cuda):
    model.eval()

    # Meters
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for batch_idx, modality_data in enumerate(test_loader):

            m_data = modality_data[1]
            labels = modality_data[0]

            if cuda:
                m_data = m_data.cuda()
                labels = labels.cuda()

            # Forward
            log_ps = model(m_data)

            # Losses
            test_loss += criterion(log_ps, labels)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))


    test_loss /= len(test_loader.dataset)

    print("Test Loss: {:.3f}.. ".format(test_loss),
          "Test Accuracy: {:.3f}".format(100* accuracy / len(test_loader)))

    return test_loss / len(test_loader), 100. * accuracy / len(test_loader)


@ex.capture
def train(_config, _run):

    # Read configs
    training_config = _config['training']
    device = torch.device("cuda" if _config['gpu']['cuda'] else "cpu")

    # Set seeds
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(training_config['seed'])
    np.random.seed(training_config['seed'])
    random.seed(training_config['seed'])
    torch.cuda.manual_seed(training_config['seed'])

    # Create Classifier
    model = Image_Classifier().to(device)
    epochs = training_config['epochs']

    # Create Dataset
    dataset = MultimodalDataset(
        data_dir=exp_dir_path('../../../scenarios/multimodal_dataset/data'),
        modalities=['image'],
        batch_size=training_config['batch_size'],
        eval_samples=10,
        validation_size=0.1,
        seed=42)

    train_loader, test_loader = dataset.train_loader, dataset.val_loader

    # Training objects
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    best_loss = sys.maxsize

    for epoch in range(1, epochs + 1):

        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch, cuda= _config['gpu']['cuda'])
        val_loss, val_accuracy = test_epoch(model, test_loader, criterion, cuda= _config['gpu']['cuda'])

        _run.log_scalar('train_loss', train_loss)
        _run.log_scalar('val_loss', val_loss.item())
        _run.log_scalar('val_accuracy', val_accuracy.item())

        # Best Loss
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)
        record_checkpoint(model=model, loss=val_loss, best_loss=best_loss,
                          optimizer=optimizer, epoch=epoch, is_best=is_best)


    # Final Saving
    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'image_classifier_checkpoint.pth.tar'),
        name='image_classifier_last_checkpoint.pth.tar')
    ex.add_artifact(
        os.path.join(
            log_dir_path('trained_models'), 'best_image_classifier_model.pth.tar'),
        name='best_image_classifier_model.pth.tar')


@ex.main
def main(_config, _run):
    os.makedirs(log_dir_path('trained_models'), exist_ok=True)
    os.makedirs(log_dir_path('results'), exist_ok=True)
    train()


if __name__ == '__main__':
    ex.run_commandline()