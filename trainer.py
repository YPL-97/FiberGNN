import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from configs import Config
from sklearn.metrics import r2_score


class Trainer:
    def __init__(self, model):
        self.model = model.to(Config.device)
        self.optimizer = optim.Adam(model.parameters(), lr=Config.lr)
        self.criterion = nn.SmoothL1Loss()

    def train(self, train_loader, val_loader):
        best_loss = float('inf')
        for epoch in range(Config.epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                batch = batch.to(Config.device)
                self.optimizer.zero_grad()
                out = self.model(batch)
                target = batch.y.view(-1, 5)
                loss = self.criterion(out, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()


            val_loss = self.validate(val_loader)
            print(f"Epoch {epoch + 1}/{Config.epochs} | "
                  f"Train Loss: {total_loss:.4e} | Val Loss: {val_loss:.4e} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(),
                           os.path.join(Config.save_dir, 'best_model.pth'))

    def validate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(Config.device)
                out = self.model(batch)
                target = batch.y.view(-1, 5)
                loss = self.criterion(out, target)
                total_loss += loss.item()
        return total_loss / len(loader)

    def test(self, loader):
        self.model.eval()
        total_loss = 0
        predictions, truths = [], []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(Config.device)
                out = self.model(batch)
                target = batch.y.view(-1, 5)

                predictions.append(out.cpu().numpy())
                truths.append(target.cpu().numpy())

                loss = self.criterion(out, target)
                total_loss += loss.item()

        predictions = np.concatenate(predictions)
        truths = np.concatenate(truths)
        mae = np.mean(np.abs(predictions - truths))
        r2 = r2_score(truths, predictions)

        return total_loss / len(loader), mae, r2