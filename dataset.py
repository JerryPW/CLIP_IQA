import os
import csv
import clip
import random
import scipy.io
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class AGIQADataset(Dataset):
    def __init__(self, root, transform, device, test=False):
        img_paths = os.path.join(root, 'images')
        mos_paths = os.path.join(root, 'data.csv')
        imgname = []
        mosall = []
        text = []
        with open(mos_paths) as f:
            reader = csv.DictReader(f)
            for row in reader:
                imgname.append(row['name'])
                mos = np.array(float(row['mos_quality'])).astype(np.float32)
                mosall.append(mos)
                text.append(row['prompt'])
        
        sample = []
        for item in range(len(mosall)):
            sample.append((os.path.join(img_paths, imgname[item]), text[item], mosall[item]))
        
        random.seed(42)
        sample = random.sample(sample, len(sample))
        
        test_size = int(len(sample) * 0.2)
        if test:
            sample = sample[:test_size]
        else:
            sample = sample[test_size:]
        
        self.samples = sample
        self.transform = transform
        self.device = device
        
    def __getitem__(self, index):
        path, text, target = self.samples[index]
        sample = self.transform(Image.open(path))
        text = clip.tokenize(text).squeeze(0)
        return {'image':sample, 'text':text, 'mos': target}

    def __len__(self):
        length = len(self.samples)
        return length
    

class AIGCIQA2023Dataset(Dataset):
    def __init__(self, root, transform, device, test=False):
        image_path = os.path.join(root, 'Image', 'allimg')
        
        mos_path = os.path.join(root, 'DATA', 'MOS', 'mosz2.mat')
        mosall = scipy.io.loadmat(mos_path)['MOSz']/20
        text_path = os.path.join(root, 'AIGCIQA2023_Prompts.csv')
        text = []
        with open(text_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for i in range(4):
                    text.append(row['prompt'])
        
        image_path_list = sorted(sorted(os.listdir(image_path))[1:], key=lambda x: int(x.split('.')[0]))
        
        sample = []
        for i in range(6):
            for j in range(400):
                image_item = i*400 + j
                sample.append((os.path.join(image_path, image_path_list[image_item]), text[j], mosall[image_item][0]))
        
        random.seed(42)
        sample = random.sample(sample, len(sample))
        
        test_size = int(len(sample) * 0.2)
        if test:
            sample = sample[:test_size]
        else:
            sample = sample[test_size:]
        
        self.samples = sample
        self.transform = transform
        self.device = device
    
    def __getitem__(self, index):
        path, text, target = self.samples[index]
        sample = self.transform(Image.open(path))
        text = clip.tokenize(text).squeeze(0)
        return {'image':sample, 'text':text, 'mos': target.astype(np.float32)}

    def __len__(self):
        length = len(self.samples)
        return length
        
                
                
            
        
        