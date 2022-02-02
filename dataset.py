import os
from pathlib import Path

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

import skimage.io as io
from skimage.transform import resize



class AbnormalDataset(Dataset):
    """
    Dataset 재정의
    
    Arg:
        data_path: The directory path where the data is stored.
        split: train or test
        clip_len: The number of clips.
    """

    def __init__(self, data_path, split='train', clip_len=16):
        self.path = Path(data_path)
        self.clip_len = clip_len
        self.split = split

        self.fnames, labels = [], []
        for dir_path in sorted(self.path.iterdir()):
            for file_path in dir_path.iterdir():
                self.fnames.append(file_path)
                labels.append(dir_path.name)

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if not os.path.exists('labels.txt'):
            with open('labels.txt', 'w') as f:
                for id, label in enumerate(sorted(self.label2index)):
                    f.writelines(str(id + 1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        buffer = self.load_frames(self.fnames[index])
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            buffer = self.randomflip(buffer)

        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)

        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def randomflip(self, buffer):
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        clip = sorted(list(Path(file_dir).glob('**/*.jpg')))
        time_index = np.random.randint(len(clip) - self.clip_len)
        clip = clip[time_index:time_index + self.clip_len]
        
        # Change the size of the image and save it in the niwifi arrangement.
        clip = np.array([resize(io.imread(str(frame)), output_shape=(112, 200), preserve_range=True).astype(np.float64) for frame in clip]).astype(np.float32)

        # center crop.
        clip = clip[:, :, 44:44 + 112, :]

        return clip


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = AbnormalDataset(data_path='/Change/Dataset/path/train', split='train', clip_len=16)
    test_data = AbnormalDataset(data_path='/Change/Dataset/path/test', split='test', clip_len=16)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break
