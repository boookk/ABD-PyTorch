import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import argparse
from tqdm import tqdm
from pathlib import Path

from model import ResNet
from ucf_dataset import AbnormalDataset


cudnn.enabled = True
cudnn.benchmark = True


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(outputs, targets):
    with torch.no_grad():
        batch_size = targets.size(0)

        _, pred = outputs.topk(1, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1))
        n_correct_elems = correct.float().sum().item()

        return n_correct_elems / batch_size


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./r3d18_KM_200ep.pth', type=str, help='pretrained model')
    parser.add_argument('--n_classes', default=2, type=int, help='num of class')
    parser.add_argument('--clip_len', default=16, type=int)
    parser.add_argument('--checkpoint', default='./checkpoint', type=Path)
    parser.add_argument('--save_epoch', default=100, type=int)
    parser.add_argument('--n_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=16, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    train_loader = DataLoader(AbnormalDataset('/DATASET/PATH/train', split='train', clip_len=16), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(AbnormalDataset('/DATASET/PATH/test', split='test', clip_len=16), batch_size=args.batch_size, pin_memory=True)

    # Load the trained model and change the FC layer.
    model = ResNet()
    model.load_state_dict(torch.load(args.model)['state_dict'])
    model.fc = nn.Linear(model.fc.in_features, args.n_classes)
    
    # Using multi gpu
    model = nn.DataParallel(model, device_ids=None).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    """ Train """
    model.train()
    for epoch in tqdm(range(args.n_epochs)):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        if epoch % args.save_epoch == args.save_epoch - 1:
            if not args.checkpoint.exists():
                args.checkpoint.mkdir()
            save_file = args.checkpoint / f'{epoch + 1}.pth'
            torch.save(model.state_dict(), save_file)
    
    """ Test """
    model.eval()
    losses = AverageMeter()
    accuracies = AverageMeter()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))

        print("[test] Epoch: {} Loss: {:.4f} Acc: {:.4f}".format(args.n_epochs, losses.avg, accuracies.avg))
