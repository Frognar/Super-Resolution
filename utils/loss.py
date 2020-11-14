import torch

from models import vgg


class SRGANPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._prepare_normalization_constants()
        self._initialize_feature_extractor()
        self._initialize_criteria()

    def _prepare_normalization_constants(self):
        self._MEAN = torch.FloatTensor([0.485, 0.456, 0.406]).cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self._STD = torch.FloatTensor([0.229, 0.224, 0.225]).cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def _initialize_feature_extractor(self):
        self._feature_extractor = vgg.TruncatedVGG19()

    def _initialize_criteria(self):
        self._content_loss_criterion = torch.nn.MSELoss().cuda()
        self._adversarial_loss_criterion = torch.nn.BCELoss().cuda()

    def forward(self, sr_discriminated, sr_images, hr_images):
        sr_feature_maps, hr_feature_maps = self._extract_feature_maps(sr_images, hr_images)

        content_loss = self._calculate_content_loss(sr_feature_maps, hr_feature_maps)
        adversarial_loss = self._calculate_adversarial_loss(sr_discriminated)
        return self._calculate_perceptual_loss(adversarial_loss, content_loss)

    def _extract_feature_maps(self, sr_images, hr_images):
        hr_images, sr_images = self._normalize_images(hr_images, sr_images)
        sr_feature_maps = self._feature_extractor(sr_images)
        hr_feature_maps = self._feature_extractor(hr_images).detach()
        return sr_feature_maps, hr_feature_maps

    def _normalize_images(self, hr_images, sr_images):
        sr_images = self._normalize_images_for_vgg(sr_images)
        hr_images = self._normalize_images_for_vgg(hr_images)
        return hr_images, sr_images

    def _normalize_images_for_vgg(self, images):
        images = self._change_value_range_to_0_1(images)
        return (images - self._MEAN) / self._STD

    @staticmethod
    def _change_value_range_to_0_1(images):
        return (images + 1.) / 2.

    def _calculate_content_loss(self, sr_feature_maps, hr_feature_maps):
        return self._content_loss_criterion(sr_feature_maps, hr_feature_maps)

    def _calculate_adversarial_loss(self, sr_discriminated):
        return self._adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))

    @staticmethod
    def _calculate_perceptual_loss(adversarial_loss, content_loss):
        rescale_factor = 1e-3
        return content_loss + rescale_factor * adversarial_loss


class ESRGANPerceptualLoss(SRGANPerceptualLoss):
    def _initialize_feature_extractor(self):
        self._feature_extractor = vgg.TruncatedVGG19BeforeActivation()

    def _initialize_criteria(self):
        self._content_loss_criterion = torch.nn.L1Loss().cuda()
        self._adversarial_loss_criterion = torch.nn.BCEWithLogitsLoss().cuda()
