import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import ImageNetDataset
from nn.model import DenseGenerator
from trainers import NetTrainer
from utils import Converter


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = DenseGenerator().to(device)
    criteria = MSELoss().to(device)
    optimizer = Adam(
        params=filter(lambda p: p.requires_grad, net.parameters()),
        lr=1e-4
    )
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        patience=3,
        factor=0.5,
        verbose=True
    )

    dataset = ImageNetDataset(
        json_path='data/train.json',
        converter=Converter()
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        shuffle=True
    )

    trainer = NetTrainer(
        net=net,
        criterion=criteria,
        optimizer=optimizer,
        scheduler=scheduler,
        data_loader=data_loader,
        device=device
    )
    trainer.train(
        max_epochs=5,
        save_path='data/checkpoints/SRDenseResNet.pth.tar'
    )


if __name__ == '__main__':
    main()
