import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, InterpolationMode, ToTensor
import os
from PIL import Image
import numpy as np

data_transforms = Compose([Resize(size=(224, 224), interpolation=InterpolationMode.NEAREST), ToTensor()])


def trans_square(image):
    img = image.convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img_h, img_w, img_c = img.shape
    if img_h != img_w:
        long_side = max(img_w, img_h)
        short_side = min(img_w, img_h)
        loc = abs(img_w - img_h) // 2
        img = img.transpose((1, 0, 2)) if img_w < img_h else img
        background = np.zeros((long_side, long_side, img_c), dtype=np.uint8)
        background[loc:loc + short_side] = img[...]
        img = background.transpose((1, 0, 2))

    return Image.fromarray(img, 'RGB')


class arthorsis_dataset(Dataset):
    def __init__(self, mode, path):
        super().__init__()

        self.data_list = []
        self.mode = mode
        if self.mode == 'train':
            file_path = os.path.join(path, "train.txt")
        elif self.mode == 'val':
            file_path = os.path.join(path, "val.txt")

        with open(file_path, 'r') as f:
            for line in f:
                img_path, level = line.split()
                self.data_list.append([img_path, int(level)])

    def __getitem__(self, item):
        img_path, label = self.data_list[item]
        img = Image.open(img_path)
        img = trans_square(img)
        img = img.convert('L')
        img = data_transforms(img)
        label = torch.tensor(label)
        return img, label

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    data = arthorsis_dataset(mode="train", path=r"D:\20230705\datasets\arthrosis\MCP")
    train_loader = DataLoader(data, batch_size=10, shuffle=True)

    for x, y in train_loader:
        print(x.shape)
        print(y.shape)
        exit()
