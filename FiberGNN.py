import os
import torch
from torch_geometric.loader import DataLoader
from configs import Config
from dataloader import split_dataset, load_data
from models import GCNModel, GATModel, GINModel, GraphSAGEModel
from trainer import Trainer
from utils import set_seed
from visualizer import visualize_test_results


if __name__ == "__main__":
    set_seed(seed=42)
    graphs = load_data()
    train_data, val_data, test_data = split_dataset(graphs)

    train_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=Config.batch_size)
    test_loader = DataLoader(test_data, batch_size=Config.batch_size)

    model = GCNModel(
        num_node_features=9,
        num_graph_attributes=5,
        output_dim=5
    ).to(Config.device)


    trainer = Trainer(model)
    trainer.train(train_loader, val_loader)

    model.load_state_dict(torch.load(os.path.join(Config.save_dir, 'best_model.pth')))
    trainer.validate(test_loader)
    test_loss, test_mae, test_r2 = trainer.test(test_loader)
    print(f"\nFinal Test Performance:")
    print(f"Loss: {test_loss:.4f} | MAE: {test_mae:.4f} | R²: {test_r2:.4f}")
    visualize_test_results(model, test_loader)
