import torch
from torchvision.transforms import transforms
from torchvision.transforms import functional as F


class Compose(transforms.Compose):
    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(transforms.ToTensor):
    def randomize_parameters(self):
        pass


class Normalize(transforms.Normalize):
    def randomize_parameters(self):
        pass


class Resize(transforms.Resize):
    def randomize_parameters(self):
        pass


class CenterCrop(transforms.CenterCrop):
    def randomize_parameters(self):
        pass


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p)
        self.randomize_parameters()

    def forward(self, img):
        if self.random_p < self.p:
            return F.hflip(img)
        return img

    def randomize_parameters(self):
        self.random_p = torch.rand(1)


class RandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=F.InterpolationMode.BILINEAR,
    ):
        super().__init__(size, scale, ratio, interpolation)
        self.randomize_parameters()

    def forward(self, img):
        if self.randomize:
            self.random_crop = self.get_params(img, self.scale, self.ratio)
            self.randomize = False

        i, j, h, w = self.random_crop
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def randomize_parameters(self):
        self.randomize = True
