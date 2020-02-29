import random
import numpy as np
import preprocess
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path, scale):
    loaded_image = Image.open(path).convert('RGB')
    original_width, original_height = loaded_image.size
    scaled_width = original_width//scale[1]
    scaled_height = original_height//scale[0]
    if(scale[0] > 1 and scale[1] > 1):
        loaded_image = transforms.Resize((scaled_height,scaled_width),2)(loaded_image)

    print(loaded_image.size)

    return loaded_image


def disparity_loader(path):
    return np.load(path).astype(np.float32)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader, scale=[1,1]):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.scale = scale

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        left_img = self.loader(left, self.scale)
        right_img = self.loader(right, self.scale)
        dataL = self.dploader(disp_L)

        if self.training:
            w, h = left_img.size
            th, tw = 128, 256

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            print(left_img.size)
            right_img = processed(right_img)

        else:
            w, h = left_img.size

            # left_img = left_img.crop((w - 1232, h - 368, w, h))
            # right_img = right_img.crop((w - 1232, h - 368, w, h))
            left_img = left_img.crop((w - 1200, h - 352, w, h))
            right_img = right_img.crop((w - 1200, h - 352, w, h))
            w1, h1 = left_img.size

            # dataL1 = dataL[h - 368:h, w - 1232:w]
            dataL = dataL[h - 352:h, w - 1200:w]

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

        dataL = torch.from_numpy(dataL).float()
        return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
