import torch
from torch.nn import BCELoss, MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import ImageNetDataset
from nn.feature_extractor import TruncatedVgg
from nn.loss import DiscriminatorLoss, PerceptualLoss
from nn.model import Discriminator, Generator
from trainers import GANTrainer
from utils import Converter


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    g_net = Generator().to(device)
    g_criterion = PerceptualLoss(
        feature_extractor=TruncatedVgg(),
        content_criterion=MSELoss(),
        adversarial_criterion=BCELoss()
    ).to(device)
    g_optimizer = Adam(
        params=filter(lambda p: p.requires_grad, g_net.parameters()),
        lr=1e-4
    )
    g_scheduler = ReduceLROnPlateau(
        optimizer=g_optimizer,
        factor=0.5,
        patience=3,
        verbose=True
    )

    d_net = Discriminator().to(device)
    d_criterion = DiscriminatorLoss(criterion=BCELoss()).to(device)
    d_optimizer = Adam(
        params=filter(lambda p: p.requires_grad, d_net.parameters()),
        lr=1e-4
    )
    d_scheduler = ReduceLROnPlateau(
        optimizer=d_optimizer,
        factor=0.5,
        patience=3,
        verbose=True
    )

    converter = Converter()
    dataset = ImageNetDataset(
        json_path='data/train.json',
        converter=converter
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        shuffle=True
    )

    trainer = GANTrainer(
        g_net=g_net,
        g_criterion=g_criterion,
        g_optimizer=g_optimizer,
        g_scheduler=g_scheduler,
        d_net=d_net,
        d_criterion=d_criterion,
        d_optimizer=d_optimizer,
        d_scheduler=d_scheduler,
        data_loader=data_loader,
        device=device
    )
    trainer.train(
        max_epochs=5,
        save_path='data/checkpoints/SRGAN.pth.tar'
    )


if __name__ == '__main__':
    main()
