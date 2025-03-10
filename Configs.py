import os
import torch

class Config:
    seed = 42
    data_path = "dataset/graph_*.csv"
    save_dir = "saved_models_gcn"
    batch_size = 128
    hidden_dim = 256
    epochs = 50
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
os.makedirs(config.save_dir, exist_ok=True)