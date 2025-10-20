import pandas as pd
import torch
import numpy as np
import glob, random
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, SAGEConv, GATConv, global_mean_pool
import matplotlib.pyplot as plt
import networkx as nx
import torch
import pickle
import os

def set_seed(seed):
    random.seed(seed)  # Python内置的random库
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # torch CPU
    torch.cuda.manual_seed(seed)  # torch GPU
    torch.cuda.manual_seed_all(seed)  # 所有GPU
    torch.backends.cudnn.deterministic = True  # 确保结果可重复
    torch.backends.cudnn.benchmark = False  # 禁用cuDNN自动优化


set_seed(42)

# 设置设备为GPU（如果可用）或CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义文件模式
graph_file_pattern = "dataset/graph_*.csv"
parameter_file_pattern = "dataset/parameter_*.csv"

# 初始化编码器和归一化器
shape_encoder = OneHotEncoder(sparse_output=False)
material_encoder = OneHotEncoder(sparse_output=False)

node_feature_scaler = MinMaxScaler()
graph_attribute_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# 强制编码器识别所有可能类别
shape_encoder.fit(np.array(['circle', 'square', 'ellipse']).reshape(-1, 1))
material_encoder.fit(np.array(['SiO2', 'ZBLAN', 'GeO2']).reshape(-1, 1))


def create_graph_dataset(graph_file_pattern, parameter_file_pattern, device=device):
    structure_files = [f for f in glob.glob(graph_file_pattern)]
    parameter_files = [f for f in glob.glob(parameter_file_pattern)]
    # 用于全局收集数据
    all_node_features, all_graph_attributes, all_targets = [], [], []

    # 遍历文件并提取原始数据
    for struct_file, param_file in zip(structure_files, parameter_files):
        # 读取结构文件
        df_structure = pd.read_csv(struct_file)
        node_features = df_structure.iloc[:, [0, 1, 2, 3]].values
        node_shapes = df_structure.iloc[:, 4].astype(str).values.reshape(-1, 1)
        node_shapes_encoded = shape_encoder.transform(node_shapes)
        node_features = np.hstack((node_features, node_shapes_encoded))
        all_node_features.append(node_features)

        # 读取参数文件
        df_params = pd.read_csv(param_file)
        graph_attributes = df_params.iloc[:, [0, 2, 3]].values
        fiber_material = df_params.iloc[:, 1].astype(str).values.reshape(-1, 1)
        fiber_material_encoded = material_encoder.transform(fiber_material)
        graph_attributes = np.hstack((graph_attributes, fiber_material_encoded))
        all_graph_attributes.append(graph_attributes)

        # 提取目标
        targets = df_params.iloc[:, 4:].values
        all_targets.append(targets)

    # 全局归一化
    all_node_features = np.vstack(all_node_features)
    node_feature_scaler.fit(all_node_features)

    all_graph_attributes = np.vstack(all_graph_attributes)
    graph_attribute_scaler.fit(all_graph_attributes)

    all_targets = np.vstack(all_targets)
    target_scaler.fit(all_targets)

    # 构建图数据集
    all_graphs = []
    for struct_file, param_file in zip(structure_files, parameter_files):
        # 处理结构文件
        df_structure = pd.read_csv(struct_file)
        node_features = df_structure.iloc[:, [0, 1, 2, 3]].values
        node_shapes = df_structure.iloc[:, 4].astype(str).values.reshape(-1, 1)
        node_shapes_encoded = shape_encoder.transform(node_shapes)
        node_features = np.hstack((node_features, node_shapes_encoded))
        node_features_normalized = node_feature_scaler.transform(node_features)
        node_features_tensor = torch.tensor(node_features_normalized, dtype=torch.float).to(device)

        # 处理参数文件
        df_params = pd.read_csv(param_file)
        graph_attributes = df_params.iloc[:, [0, 2, 3]].values
        fiber_material = df_params.iloc[:, 1].astype(str).values.reshape(-1, 1)
        fiber_material_encoded = material_encoder.transform(fiber_material)
        graph_attributes = np.hstack((graph_attributes, fiber_material_encoded))
        graph_attributes_normalized = graph_attribute_scaler.transform(graph_attributes)
        graph_attributes_tensor = torch.tensor(graph_attributes_normalized, dtype=torch.float).to(device)

        # 目标归一化
        targets = df_params.iloc[:, 4:].values
        targets_normalized = target_scaler.transform(targets)
        targets_tensor = torch.tensor(targets_normalized, dtype=torch.float).to(device)

        # 无边图
        edge_index = torch.empty((2, 0), dtype=torch.long).to(device)

        for i in range(graph_attributes_tensor.shape[0]):
            graph = Data(
                x=node_features_tensor,
                edge_index=edge_index,
                y=targets_tensor[i],
                graph_attr=graph_attributes_tensor[i]
            )
            all_graphs.append(graph)

    return all_graphs


# 数据集划分
def split_dataset(graphs):
    train_data, test_data = train_test_split(graphs, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
    return train_data, val_data, test_data


all_graph_data = create_graph_dataset(graph_file_pattern, parameter_file_pattern)

train_graphs, val_graphs, test_graphs = split_dataset(all_graph_data)
print(f"Train graphs: {len(train_graphs)}, Val graphs: {len(val_graphs)}, Test graphs: {len(test_graphs)}")


# GNN Backbone
class GINModel(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, num_graph_attributes):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(nn.Sequential(nn.Linear(num_node_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.fc1 = nn.Linear(hidden_dim + num_graph_attributes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch, graph_attr):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # (num_graphs, hidden_dim)
        graph_attr = graph_attr.view(-1, num_graph_attributes)    # 将 graph_attr 变为形状为 [batch_size, num_graph_attributes]
        x = torch.cat((x, graph_attr), dim=1)  # x 的形状为 [batch_size, hidden_dim + num_graph_attributes]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, num_epochs, final_model_path):
    train_loss_list = []
    test_loss_list = []
    best_test_loss = float('inf')

    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch, batch.graph_attr)
            target = batch.y.view(-1, 5)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 计算训练集的平均 loss
        avg_train_loss = train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        # 评估模式
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch, batch.graph_attr)
                target = batch.y.view(-1, 5)
                loss = criterion(out, target)
                test_loss += loss.item()

        # 计算测试集的平均 loss
        avg_test_loss = test_loss / len(test_loader)
        test_loss_list.append(avg_test_loss)

        # 打印当前 epoch 的训练和测试 loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            # 保存当前最佳模型的 state_dict
            torch.save(model.state_dict(), 'best_model1.pth')
            print(f"Epoch {epoch + 1}: Test loss improved to {avg_test_loss:.4f}, saving model...")
        else:
            print(f"Epoch {epoch + 1}: Test loss did not improve ({avg_test_loss:.4f})")

    # 在训练结束后重新加载最佳模型（加载 state_dict）
    best_model = GINModel(
        num_node_features=train_graphs[0].num_node_features,
        hidden_dim=model.fc1.in_features - 6,  # 根据 fc1 输入特征计算 hidden_dim
        output_dim=model.fc2.out_features,
        num_graph_attributes=6
    ).to(device)
    best_model.load_state_dict(torch.load('best_model1.pth', map_location=device))
    best_model.to(device)

    # 保存最终的模型的 state_dict
    torch.save(best_model.state_dict(), final_model_path)

    # 绘制损失曲线（在所有 epoch 结束后）
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_loss_list, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), test_loss_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss over Epochs')
    plt.legend()
    plt.show()

# 定义预处理器保存目录
preprocessors_dir = 'preprocessors'
os.makedirs(preprocessors_dir, exist_ok=True)

# 保存编码器和归一化器
with open(os.path.join(preprocessors_dir, 'shape_encoder.pkl'), 'wb') as f:
    pickle.dump(shape_encoder, f)

with open(os.path.join(preprocessors_dir, 'material_encoder.pkl'), 'wb') as f:
    pickle.dump(material_encoder, f)

with open(os.path.join(preprocessors_dir, 'node_feature_scaler.pkl'), 'wb') as f:
    pickle.dump(node_feature_scaler, f)

with open(os.path.join(preprocessors_dir, 'graph_attribute_scaler.pkl'), 'wb') as f:
    pickle.dump(graph_attribute_scaler, f)

with open(os.path.join(preprocessors_dir, 'target_scaler.pkl'), 'wb') as f:
    pickle.dump(target_scaler, f)

print("编码器和归一化器已保存。")


def validate(model, val_loader, criterion):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch, batch.graph_attr)  # 使用图属性

            # 确保目标的形状与输出匹配
            target = batch.y.view(-1, 5)
            if target.shape[0] != out.shape[0]:
                print(f"Warning: target size {target.shape} does not match output size {out.shape}.")
                continue

            loss = criterion(out, target)
            valid_loss += loss.item()

    avg_valid_loss = valid_loss / len(val_loader)
    print(f"Validation Loss: {avg_valid_loss}")


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch, batch.graph_attr)  # 使用图属性

            # 确保目标的形状与输出匹配
            target = batch.y.view(-1, 5)
            if target.shape[0] != out.shape[0]:
                print(f"Warning: target size {target.shape} does not match output size {out.shape}.")
                continue

            loss = criterion(out, target)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss}")


# 假设 train_graphs, val_graphs, test_graphs 已经定义

num_graph_attributes = 6
input_dim = train_graphs[0].num_node_features + num_graph_attributes

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

# 定义保存模型的路径
best_model_path = "best_model1.pth"
final_model_path = 'final_model1.pth'

# 创建模型实例
model = GINModel(num_node_features=train_graphs[0].num_node_features,
                 hidden_dim=64,
                 output_dim=5,  # 假设输出维度为 5
                 num_graph_attributes=6).to(device)

# 创建优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 定义损失函数
criterion = nn.SmoothL1Loss()

# 训练并评估模型
train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, num_epochs=1,
                   final_model_path=final_model_path)

# 验证和测试
validate(model, val_loader, criterion)
test(model, test_loader, criterion)


