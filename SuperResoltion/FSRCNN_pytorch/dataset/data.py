from os.path import exists, join, basename
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from .dataset import DatasetFromFolder


def image_dir():
    return join("./", "dataset/data/DIV2K")


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        CenterCrop(crop_size),
        Resize(crop_size // upscale_factor),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        CenterCrop(crop_size),
        ToTensor(),
    ])


# train
def get_training_set(upscale_factor):
    root_dir = image_dir()
    train_dir = join(root_dir, "DIV2K_train_HR")  # train path
    crop_size = calculate_valid_crop_size(128, upscale_factor)
    print(crop_size)
    return DatasetFromFolder(train_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))


# test
def get_test_set(upscale_factor):
    root_dir = image_dir()
    test_dir = join(root_dir, "DIV2K_valid_HR")  # test path
    crop_size = calculate_valid_crop_size(128, upscale_factor)

    return DatasetFromFolder(test_dir,
                             input_transform=input_transform(crop_size, upscale_factor),
                             target_transform=target_transform(crop_size))
