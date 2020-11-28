import torch


_mean = torch.FloatTensor([0.485, 0.456, 0.406])
_std = torch.FloatTensor([0.229, 0.224, 0.225])


class PerceptualLoss:
    def __init__(self, vgg, content_criterion, adversarial_criterion):
        super().__init__()
        self.vgg = vgg
        self.content_criterion = content_criterion
        self.adversarial_criterion = adversarial_criterion
        self.mean = _mean.unsqueeze(0).unsqueeze(2)
        self.std = _std.unsqueeze(0).unsqueeze(2)

    def move_to_cuda(self):
        self.vgg = self.vgg.cuda()
        self.content_criterion = self.content_criterion.cuda()
        self.adversarial_criterion = self.adversarial_criterion.cuda()
        self.mean = _mean.cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.std = _std.cuda().unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def move_to_cpu(self):
        self.vgg = self.vgg.cpu()
        self.content_criterion = self.content_criterion.cpu()
        self.adversarial_criterion = self.adversarial_criterion.cpu()
        self.mean = _mean.unsqueeze(0).unsqueeze(2)
        self.std = _std.unsqueeze(0).unsqueeze(2)

    def calculate(self, labels, generated, real):
        content_loss = self.content_loss_of(*self.extracted(generated, real))
        adversarial_loss = self.adversarial_loss(labels)

        rescale_factor = 1e-3
        return content_loss + rescale_factor * adversarial_loss

    def extracted(self, generated, real):
        generated, real = self.normalize(generated, real)
        return self.vgg(generated), self.vgg(real).detach()

    def normalize(self, generated, real):
        return self.normalize_for_vgg(generated), self.normalize_for_vgg(real)

    def normalize_for_vgg(self, tensor):
        return (((tensor + 1.) / 2.) - self.mean) / self.std

    def content_loss_of(self, generated_features, real_features):
        return self.content_criterion(generated_features, real_features)

    def adversarial_loss(self, labels):
        return self.adversarial_criterion(labels, torch.ones_like(labels))
