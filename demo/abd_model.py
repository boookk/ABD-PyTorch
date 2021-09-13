import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import transforms

from PIL import Image
import numpy as np

from model import generate_model


class abd_model():
    def __init__(self):
        self.n_classes = 2
        self.model_path = '/home/bobo/ABD-PyTorch/checkpoint/30.pth'
        self.class_name = self.get_class()
        self.model = self.get_model()
        self.transform = self.get_transform()
        self.clip = []

    def get_model(self):
        model = generate_model(n_classes=self.n_classes)
        
        load_params = torch.load(self.model_path)
        new_params = model.state_dict().copy()
        for j in load_params:
            jj = j.split('.')
            if jj[0] == 'module':
                new_params['.'.join(jj[1:])] = load_params[j]
            else:
                new_params[j] = load_params[j]
        model.load_state_dict(new_params)

        if torch.cuda.device_count() > 0 and torch.cuda.is_available():
            model = nn.DataParallel(model).cuda()

        return model

    def get_class(self):
        ############################################
        class_names = ['normal', 'swoon']
        ###########################################
        return class_names

    def get_transform(self):
        transform = transforms.Compose([
            transforms.Resize(244),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return transform

    def preprocessing(self, clip):
        clip = [self.transform(Image.fromarray(np.uint8(img)).convert('RGB')) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        clip = torch.stack((clip,), 0)
        return clip

    def relation_recognition(self):
        if len(self.clip) == 16:
            self.model.eval()
            clip = self.preprocessing(self.clip)
            with torch.no_grad():
                predict = self.model(clip)
                predict = F.softmax(predict, dim=1).cpu()
                score, class_predict = torch.max(predict, 1)
            return score[0], self.class_name[class_predict[0]]
        return None, None

    def save_clip(self, img):
        self.clip.append(img)
        if len(self.clip) > 16:
            del self.clip[0]
