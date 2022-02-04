import torchvision.transforms as transforms
import augly.image as imaugs
from PIL import Image

COLOR_JITTER_PARAMS = {
    "brightness_factor": 1.2,
    "contrast_factor": 0.4,
    "saturation_factor": 0.2,
    "p": 0.8
}

AUGMENTATIONS = [
    imaugs.Blur(),
    #imaugs.ColorJitter(**COLOR_JITTER_PARAMS),
    imaugs.transforms.HFlip(p=0.5),
    #imaugs.RandomNoise(mean = 0.0, var = 0.1, seed = 42, p = 0.2),
    imaugs.transforms.Contrast(factor = 1.7),
    imaugs.Brightness(1.5)

]

TRANSFORMS = imaugs.Compose(AUGMENTATIONS)
TENSOR_TRANSFORMS = transforms.Compose(AUGMENTATIONS + [transforms.ToTensor()])