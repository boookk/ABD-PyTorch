import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
from pathlib import Path
from itertools import chain


class AbnormalDataset(Dataset):
    def __init__(self, path, frame_size, size):
        self.path = Path(path).absolute()
        self.frame_size = frame_size
        self.size = size
        self.class_path = list(self.path.iterdir())
        self.data_ = list(chain(*[c.iterdir() for c in self.class_path]))
        self.data = [d for d in self.data_ if len(list(d.glob('**/*.png'))) > self.frame_size]
        self.spatial_transforms = self.get_spatial_transforms()
        self.classes = ['normal', 'swoon']

    def get_temporal_transforms(self, s):
        middle = s // 2
        half = self.frame_size // 2
        start = middle - half
        end = middle + half
        return start, end

    def get_spatial_transforms(self):
        transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return transform

    def __getitem__(self, idx):
        target = self.classes.index(str(self.data[idx].parent).split('/')[-1])
        img_list = list(self.data[idx].glob('**/*.png'))
        start, end = self.get_temporal_transforms(len(img_list))

        clip = []
        for img_path in img_list[start:end]:
            img = Image.open(img_path).convert('RGB')
            img = self.spatial_transforms(img)
            clip.append(img)
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        return clip, target

    def __len__(self):
        return len(self.data)
