import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from dataset import ImageNetDataset
from nn.model import Generator
from trainers import NetTrainer
from utils import Converter, parse_training_args


def main():
    args = parse_training_args("SRResNet")
    epochs = args.epochs
    load_path = args.load
    out_path = args.out
    cuda = args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')

    net = Generator().to(device)
    criteria = MSELoss().to(device)
    optimizer = Adam(params=filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    dataset = ImageNetDataset(json_path='data/train.json', converter=Converter())
    data_loader = DataLoader(dataset=dataset, batch_size=16, num_workers=4,
                             pin_memory=True, shuffle=True)

    trainer = NetTrainer(net=net, criterion=criteria, optimizer=optimizer,
                         scheduler=scheduler, data_loader=data_loader, device=device)

    if load_path:
        trainer.load(load_path)

    trainer.train(max_epochs=epochs, save_path=out_path)


if __name__ == '__main__':
    main()
