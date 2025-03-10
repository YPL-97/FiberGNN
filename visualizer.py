import os
import torch
import numpy as np
import pandas as pd
from configs import Config
import matplotlib.pyplot as plt
from dataloader import processor

def visualize_test_results(model, test_loader):
    best_model_path = os.path.join(Config.save_dir, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path))
    model.to(Config.device)
    model.eval()

    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(Config.device)
            out = model(batch)
            all_targets.append(batch.y.cpu().numpy())
            all_predictions.append(out.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    target_scaler = processor.target_scaler
    num_features = target_scaler.scale_.shape[0]

    if all_targets.ndim == 1 or all_targets.shape[1] != num_features:
        all_targets = target_scaler.inverse_transform(all_targets.reshape(-1, num_features))
        all_predictions = target_scaler.inverse_transform(all_predictions.reshape(-1, num_features))
    else:
        all_targets = target_scaler.inverse_transform(all_targets)
        all_predictions = target_scaler.inverse_transform(all_predictions)

    param_names = [
        "Effective Refractive Index (a.u.)",
        "Effective Mode Area (um$^2$)",
        "Nonlinear Coefficient (1/W/km)",
        "Dispersion Coefficient (ps/nm/km)",
        "Group velocity dispersion (ps$^2$/km)" ]

    df = pd.DataFrame({
        'Predictions (1)': all_targets[:, 0],
        'Ground Truths (1)': all_predictions[:, 0],
        'Predictions (2)': all_targets[:, 1],
        'Ground Truths (2)': all_predictions[:, 1],
        'Predictions (3)': all_targets[:, 2],
        'Ground Truths (3)': all_predictions[:, 2],
        'Predictions (4)': all_targets[:, 3],
        'Ground Truths (4)': all_predictions[:, 3],
        'Predictions (5)': all_targets[:, 4],
        'Ground Truths (5)': all_predictions[:, 4],
    })

    plt.rcParams['font.family'] = 'Times New Roman'
    for i in range(all_targets.shape[1]):
        plt.figure(figsize=(8, 6))
        plt.scatter(all_targets[:, i], all_predictions[:, i], alpha=0.5)
        plt.plot([all_targets[:, i].min(), all_targets[:, i].max()],
                 [all_targets[:, i].min(), all_targets[:, i].max()],
                 'r--')
        plt.title(f'{param_names[i]}', fontsize=20)
        plt.xlabel('FEM', fontsize=20)
        plt.ylabel('GNN', fontsize=20)
        plt.xlim(all_targets[:, i].min(), all_targets[:, i].max())
        plt.ylim(all_targets[:, i].min(), all_targets[:, i].max())

        plt.tick_params(axis='both', labelsize=20, direction='in')
        plt.grid(False)
        plt.show()

