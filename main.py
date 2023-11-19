import torch
import argparse
from IQAtrainer import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="cfg.yml",
                        type=str, help="config file path")
    parser.add_argument("--mode", default="train", type=str,
                        help="mode for train/test")
    args = parser.parse_args()
    
    torch.set_default_dtype(torch.float32)

    if args.mode == 'train':
        trainer = Trainer(args.cfg)
        trainer.train()
    elif args.mode == 'test':
        trainer = Trainer(args.cfg, mode='test')
        trainer.test()