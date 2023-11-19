import torch
import torch.nn as nn

class IQAMLPModel(nn.Module):
    def __init__(self, cfg):
        super(IQAMLPModel, self).__init__()
        input_dim = cfg['input_dim']
        hidden_dim = cfg['hidden_dim']
        output_dim = cfg['output_dim']
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, image_features, text_features):
        features = torch.cat((image_features, text_features), dim=1)
        output = self.mlp(features)

        return output
    
    
class IQADecoderModel(nn.Module):
    def __init__(self, cfg):
        super(IQADecoderModel, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        

    def forward(self, image_features, text_features):
        features = torch.cat((image_features, text_features), dim=1)
        output = self.decoder(features)

        return output