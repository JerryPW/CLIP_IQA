import os
import yaml
import torch
import itertools
import numpy as np
from tqdm import tqdm, trange
import clip
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy import stats
from dataset import AGIQADataset, AIGCIQA2023Dataset
from IQAmodel import IQAMLPModel, IQADecoderModel

from torch.utils.tensorboard import SummaryWriter

class Trainer(object):
    """Trainer object for endoscopy reconstruction.
    """
    def __init__(self, cfg_dir, mode="train"):
        with open(cfg_dir, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        
        self.cfg = cfg
        
        # load clip model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clipmodel, preprocess = clip.load("ViT-B/32", device=self.device)
        
        # build dataset
        if cfg['data']['info_dir'] == 'AGIQA-3K':
            self.train_dataset = AGIQADataset('AGIQA-3K', preprocess, self.device, test=False)
            self.test_dataset = AGIQADataset('AGIQA-3K', preprocess, self.device, test=True)
        elif cfg['data']['info_dir'] == 'AIGCIQA2023':
            self.train_dataset = AIGCIQA2023Dataset('AIGCIQA2023', preprocess, self.device, test=False)
            self.test_dataset = AIGCIQA2023Dataset('AIGCIQA2023', preprocess, self.device, test=True)
        else:
            raise NotImplementedError
        
        # build IQAmodel
        self.IQAmodel = IQADecoderModel(cfg['IQAmodel'])
        
        self.IQAmodel.to(self.device)
        self.clipmodel.to(self.device)
        
        # build dataset loader
        self.train_loader = DataLoader(self.train_dataset, batch_size=cfg['train']['batch'], shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=cfg['train']['batch'], shuffle=True)
        
        
        self.optimizer = optim.Adam(self.IQAmodel.parameters(), lr=cfg['train']['lr'])
        self.criterion = nn.MSELoss()
        self.use_ranking = cfg['train']['use_ranking_loss']
        self.ranking_weight = cfg['train']['ranking_loss_weight']
        self.coo_combs = list(itertools.combinations(range(self.cfg['train']['batch']), 2))
        
        self.num_epochs = cfg['train']['iteration']
        self.start_epoch = 0
        
        if mode == 'test':
            self.load_checkpoints(self.num_epochs)
        
        if cfg['exp']['load_checkpoint']:
            self.load_checkpoints()
        
    
    def train(self):
        writer = SummaryWriter(os.path.join(self.cfg['exp']['exp_dir'], 'summaries', self.cfg['exp']['exp_name']))
        
        for epoch in trange(self.start_epoch, self.num_epochs, desc="TRAIN|"):
            self.IQAmodel.train()
            total_loss = 0.0

            pred_scores = []
            gt_scores = []
            
            for batch in self.train_loader:
                image = batch['image'].to(self.device)
                text = batch['text'].to(self.device)
                labels = batch['mos'].to(self.device)
                
                batch_size = image.shape[0]
    
                with torch.no_grad():
                    image_features = self.clipmodel.encode_image(image).float()
                    text_features = self.clipmodel.encode_text(text).float()

                self.optimizer.zero_grad()
                outputs = self.IQAmodel(image_features, text_features).squeeze(1)
                
                pred_scores = pred_scores + outputs.cpu().tolist()
                gt_scores = gt_scores + labels.cpu().tolist()
                
                rmse_loss = self.criterion(outputs, labels)
                
                ranking_loss = 0
                if self.use_ranking:
                    print("here")
                    if batch_size == self.cfg['train']['batch']:
                        coo_combs = torch.tensor(self.coo_combs).long()
                    else:
                        coo_combs = torch.tensor(list(itertools.combinations(range(batch_size), 2)))
                    
                    pred = outputs[coo_combs]
                    target = labels[coo_combs]
                    pred_rank = pred[target[..., 0] > target[..., 1]]
                    pred_wrong = pred_rank[pred_rank[..., 0] < pred_rank[..., 1]]
                    ranking_loss += torch.mean(pred_wrong[..., 1] - pred_wrong[..., 0])
                
                loss = rmse_loss + self.ranking_weight * ranking_loss
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            average_loss = total_loss / len(self.train_loader)
            train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
            train_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
            tqdm_txt = f"Epoch [{epoch + 1}/{self.num_epochs}] - Loss: {average_loss:.4f}  SRCC: {train_srcc}  PLCC: {train_plcc}"
            tqdm.write(tqdm_txt)
            
            writer.add_scalar('loss', average_loss, epoch)
            writer.add_scalar('SRCC', train_srcc, epoch)
            writer.add_scalar('PLCC', train_plcc, epoch)
            
            if (epoch+1) % self.cfg['train']['save_freq'] == 0:
                self.save_checkpoints(self.num_epochs)
        self.test()
            
        
    def test(self):
        total_loss = 0.0
        pred_scores = []
        gt_scores = []
        for batch in self.test_loader:
            image = batch['image'].to(self.device)
            text = batch['text'].to(self.device)
            labels = batch['mos'].to(self.device)

            with torch.no_grad():
                image_features = self.clipmodel.encode_image(image).float()
                text_features = self.clipmodel.encode_text(text).float()

            self.optimizer.zero_grad()
            outputs = self.IQAmodel(image_features, text_features).squeeze(1)
            pred_scores = pred_scores + outputs.cpu().tolist()
            gt_scores = gt_scores + labels.cpu().tolist()
            
            loss = self.criterion(outputs, labels)
            total_loss += loss.item()

        average_loss = total_loss / len(self.train_loader)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)
        print(f"Test Loss: {average_loss:.4f}  SRCC: {test_srcc}  PLCC: {test_plcc}")
        
        # import matplotlib.pyplot as plt
        
        # slope, intercept = np.polyfit(gt_scores, pred_scores, 1)

        # # 画散点图
        # plt.scatter(gt_scores, pred_scores, marker='o', label='Scatter Plot')

        # # 画拟合直线
        # fit_line = slope * np.array(gt_scores) + intercept
        # plt.plot(gt_scores, fit_line, color='red', label='Fit Line')

        # # 添加标题和标签
        # plt.title('AIGCIQA2023_correspondence')
        # plt.xlabel('Ground Truth Scores')
        # plt.ylabel('Prediction Scores')

        # plt.savefig('AIGCIQA2023_correspondence.png')
        
    
    def save_checkpoints(self, epoch):
        print("=> Saving Checkpoint...")
        checkpoint = {
        'epoch': epoch,
        'model_state_dict': self.IQAmodel.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        os.makedirs(self.cfg['exp']['exp_dir'], exist_ok=True)
        log_dir = self.cfg['exp']['exp_dir']
        os.makedirs(os.path.join(log_dir, self.cfg['exp']['exp_name']), exist_ok=True)
        exp_dir = os.path.join(log_dir, self.cfg['exp']['exp_name'])
        checkpoint_path = os.path.join(exp_dir, 'checkpoints{:05d}'.format(epoch))

        torch.save(checkpoint, checkpoint_path)
        print("=> Checkpoint Saved in {}".format(checkpoint_path))
    
    def load_checkpoints(self, epoch = None):
        print("=> Loading Checkpoint...")
        exp_dir = os.path.join(self.cfg['exp']['exp_dir'], self.cfg['exp']['exp_name'])
        checkpoint_path_list = sorted(os.listdir(exp_dir))
        if epoch is not None:
            for checkpoint in checkpoint_path_list:
                check_point_epoch = int(checkpoint[-5:])
                if check_point_epoch == epoch:
                    checkpoint_path = checkpoint
                    break
        else:
            checkpoint_path = checkpoint_path_list[-1]
        print("=> Found Checkpoint: {}".format(checkpoint_path))
        checkpoint = torch.load(os.path.join(exp_dir, checkpoint_path))
        self.IQAmodel.load_state_dict(checkpoint['model_state_dict'])
        self.IQAmodel.to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        print("=> Checkpoint Loaded!")
        
        