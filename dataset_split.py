from torchvision.datasets import ImageFolder
from torch.utils.data import random_split

datas = ImageFolder('/home/bokyung/Downloads/image/datas')
train_size = int(0.8 * len(datas))
test_size = len(datas) - train_size
train, test = random_split(datas, [train_size, test_size])

for img, target in train:
    img.show()
    break
