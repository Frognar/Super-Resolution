import torch

from models.vgg import TruncatedVGG19


class SRGANPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.__prepare_normalization_constants()
        self.__initialize_feature_extractor()
        self.__initialize_criteria()

    def __prepare_normalization_constants(self):
        self.__MEAN = torch.FloatTensor([0.485, 0.456, 0.406]).cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.__STD = torch.FloatTensor([0.229, 0.224, 0.225]).cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def __initialize_feature_extractor(self):
        self.__feature_extractor = TruncatedVGG19()

    def __initialize_criteria(self):
        self.__content_loss_criterion = torch.nn.MSELoss().cuda()
        self.__adversarial_loss_criterion = torch.nn.BCELoss().cuda()

    def forward(self, sr_discriminated, sr_images, hr_images):
        sr_feature_maps, hr_feature_maps = self.__extract_feature_maps(sr_images, hr_images)

        content_loss = self.__calculate_content_loss(sr_feature_maps, hr_feature_maps)
        adversarial_loss = self.calculate_adversarial_loss(sr_discriminated)
        return self.__calculate_perceptual_loss(adversarial_loss, content_loss)

    def __extract_feature_maps(self, sr_images, hr_images):
        hr_images, sr_images = self.__normalize_images(hr_images, sr_images)
        sr_feature_maps = self.__feature_extractor(sr_images)
        hr_feature_maps = self.__feature_extractor(hr_images).detach()
        return sr_feature_maps, hr_feature_maps

    def __normalize_images(self, hr_images, sr_images):
        sr_images = self.__normalize_images_for_vgg(sr_images)
        hr_images = self.__normalize_images_for_vgg(hr_images)
        return hr_images, sr_images

    def __normalize_images_for_vgg(self, images):
        images = self.__change_value_range_to_0_1(images)
        return (images - self.__MEAN) / self.__STD

    @staticmethod
    def __change_value_range_to_0_1(images):
        return (images + 1.) / 2.

    def __calculate_content_loss(self, sr_feature_maps, hr_feature_maps):
        return self.__content_loss_criterion(sr_feature_maps, hr_feature_maps)

    def calculate_adversarial_loss(self, sr_discriminated):
        return self.__adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))

    @staticmethod
    def __calculate_perceptual_loss(adversarial_loss, content_loss):
        rescale_factor = 1e-3
        return content_loss + rescale_factor * adversarial_loss
