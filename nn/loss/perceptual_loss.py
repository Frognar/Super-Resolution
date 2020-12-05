import torch
import torch.nn as nn


class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor, content_criterion,
                 adversarial_criterion):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.content_criterion = content_criterion
        self.adversarial_criterion = adversarial_criterion
        self.mean = torch.FloatTensor(
            [0.485, 0.456, 0.406]
        ).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.std = torch.FloatTensor(
            [0.229, 0.224, 0.225]
        ).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.cuda_mean = self.mean.cuda()
        self.cuda_std = self.std.cuda()

    def forward(self, sr_images, hr_images, labels):
        c_loss = self.content_criterion(*self.extract(sr_images, hr_images))
        a_loss = self.adversarial_criterion(labels, torch.ones_like(labels))
        return c_loss + 1e-3 * a_loss

    def extract(self, sr_images, hr_images):
        sr_features = self.feature_extractor(self.normalize(sr_images))
        hr_features = self.feature_extractor(self.normalize(hr_images))
        return sr_features, hr_features

    def normalize(self, images):
        if images.device.type == torch.device('cuda').type:
            return (images - self.cuda_mean) / self.cuda_std
        return (images - self.mean) / self.std
