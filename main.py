import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch.backends.cudnn as cudnn

import argparse
from tqdm import tqdm
from pathlib import Path

from model import generate_model
from dataset import AbnormalDataset


cudnn.enabled = True
cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='/home/bobo/png/', type=Path, help='path of video')
    parser.add_argument('--size', default=224, type=int, help='image size')
    parser.add_argument('--n_classes', default=2, type=int, help='num of class')
    parser.add_argument('--frame_size', default=16, type=int)
    parser.add_argument('--checkpoint', default='./checkpoint', type=Path)
    parser.add_argument('--save_epoch', default=10, type=int)
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)

    return parser.parse_args()


def get_count(dataset):
    cnt = [0] * 2
    labels = []
    for i, inputs in enumerate(dataset):
        x, y = inputs
        cnt[y] += 1
        labels.append(y)

    return sum(cnt), cnt
  

if __name__ == '__main__':
    args = get_args()

    datasets = AbnormalDataset(args.data, args.frame_size, args.size)

    train_size = int(len(datasets) * 0.8)
    test_size = len(datasets) - train_size
    train_dataset, test_dataset = random_split(datasets, [train_size, test_size])

    # train, test class 별로 데이터 갯수 카운
    print(f'Train: {get_count(train_dataset)}')
    print(f'Test: {get_count(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True)

    model = generate_model(n_classes=args.n_classes)

    # Using GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    model.train()
    for epoch in tqdm(range(args.n_epochs)):
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        else:
            print('Epochs : {}/{{}\t'
                  'Training loss: {:.2f}'.format(epoch+1, args.n_epochs, running_loss / len(train_loader)))

        if epoch % args.save_epoch == args.save_epoch - 1:
            if not args.checkpoint.exists():
                args.checkpoint.mkdir()
            save_file = args.checkpoint / f'{epoch + 1}.pth'
            torch.save(model.state_dict(), save_file)

    model.eval()
    correct = 0.0
    with torch.no_grad():
        for k, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            correct += torch.sum(predicted == targets.data)

    epoch_acc = 100 * correct / len(test_loader.dataset)
    print('Test Accuracy : {:.2f}%'.format(epoch_acc))
