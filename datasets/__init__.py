from torch.utils.data import DataLoader

from datasets.datasets import TrainDataset
from utils.image_converter import Converter


def get_data_loader(image_list_path='./data/train.json',
                    crop_size=96,
                    upscale_factor=4,
                    num_workers=4,
                    batch_size=16
                    ):
    return DataLoader(
        dataset=TrainDataset(
            image_list_path=image_list_path,
            converter=Converter(
                crop_size=crop_size,
                upscale_factor=upscale_factor
            )
        ),
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )
