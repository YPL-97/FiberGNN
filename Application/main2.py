# predict.py

import pandas as pd
import torch
import numpy as np
import pickle
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
import os


# 设置随机种子（可选）
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 定义 GIN 模型
class GINModel(nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim, num_graph_attributes):
        super(GINModel, self).__init__()
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ))
        self.fc1 = nn.Linear(hidden_dim + num_graph_attributes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch, graph_attr):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # (num_graphs, hidden_dim)
        graph_attr = graph_attr.view(-1, graph_attr.size(-1))  # 保持维度为 [batch_size, num_graph_attributes]
        x = torch.cat((x, graph_attr), dim=1)  # [batch_size, hidden_dim + num_graph_attributes]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 加载编码器和归一化器
def load_preprocessors(preprocessors_dir='preprocessors'):
    with open(os.path.join(preprocessors_dir, 'shape_encoder.pkl'), 'rb') as f:
        shape_encoder = pickle.load(f)
    with open(os.path.join(preprocessors_dir, 'material_encoder.pkl'), 'rb') as f:
        material_encoder = pickle.load(f)
    with open(os.path.join(preprocessors_dir, 'node_feature_scaler.pkl'), 'rb') as f:
        node_feature_scaler = pickle.load(f)
    with open(os.path.join(preprocessors_dir, 'graph_attribute_scaler.pkl'), 'rb') as f:
        graph_attribute_scaler = pickle.load(f)
    with open(os.path.join(preprocessors_dir, 'target_scaler.pkl'), 'rb') as f:
        target_scaler = pickle.load(f)
    return shape_encoder, material_encoder, node_feature_scaler, graph_attribute_scaler, target_scaler


# 创建单个图数据用于预测
def create_single_graph_for_prediction(graph_file, parameter_file, device, shape_encoder, material_encoder,
                                       node_feature_scaler, graph_attribute_scaler, num_graph_attributes=6):
    """
    为预测创建单个图数据（节点特征和图的属性）。

    Args:
        graph_file (str): 存储图结构的文件路径（CSV 格式）。
        parameter_file (str): 存储图参数的文件路径（CSV 格式）。
        device (torch.device): 计算设备（'cpu' 或 'cuda'）。
        shape_encoder (OneHotEncoder): 形状编码器。
        material_encoder (OneHotEncoder): 材质编码器。
        node_feature_scaler (MinMaxScaler): 节点特征归一化器。
        graph_attribute_scaler (MinMaxScaler): 图属性归一化器。
        num_graph_attributes (int): 图的属性维度（默认为6）。

    Returns:
        Data: 处理后的图数据，包含节点特征、边信息和图属性。
        torch.Tensor: batch tensor。
    """
    # 检查文件是否存在
    if not os.path.isfile(graph_file):
        raise FileNotFoundError(f"Graph structure file not found: {graph_file}")
    if not os.path.isfile(parameter_file):
        raise FileNotFoundError(f"Graph parameter file not found: {parameter_file}")

    # 读取图的结构文件
    df_structure = pd.read_csv(graph_file)
    node_features = df_structure.iloc[:, [0, 1, 2, 3]].values
    node_shapes = df_structure.iloc[:, 4].astype(str).values.reshape(-1, 1)

    # 使用编码器进行形状编码
    node_shapes_encoded = shape_encoder.transform(node_shapes)
    node_features = np.hstack((node_features, node_shapes_encoded))

    # 归一化节点特征
    node_features_normalized = node_feature_scaler.transform(node_features)
    node_features_tensor = torch.tensor(node_features_normalized, dtype=torch.float).to(device)

    # 读取参数文件
    df_params = pd.read_csv(parameter_file)
    graph_attributes = df_params.iloc[:, [0, 2, 3]].values
    fiber_material = df_params.iloc[:, 1].astype(str).values.reshape(-1, 1)

    # 使用编码器进行材质编码
    fiber_material_encoded = material_encoder.transform(fiber_material)
    graph_attributes = np.hstack((graph_attributes, fiber_material_encoded))

    # 归一化图属性
    graph_attributes_normalized = graph_attribute_scaler.transform(graph_attributes)
    graph_attributes_tensor = torch.tensor(graph_attributes_normalized, dtype=torch.float).to(device)

    # 如果只有一个图，将 graph_attributes 的维度调整为 [1, num_graph_attributes]
    if graph_attributes_tensor.shape[0] == 1:
        graph_attributes_tensor = graph_attributes_tensor.unsqueeze(0)

    # 创建无边图
    edge_index = torch.empty((2, 0), dtype=torch.long).to(device)

    # 创建 batch tensor（单个图，所有节点属于同一图）
    batch = torch.zeros(node_features_tensor.size(0), dtype=torch.long).to(device)

    # 创建图数据对象
    graph = Data(
        x=node_features_tensor,
        edge_index=edge_index,
        graph_attr=graph_attributes_tensor
    ).to(device)

    return graph, batch


# 预测函数
def predict(model, graph_file, parameter_file, device, shape_encoder, material_encoder,
            node_feature_scaler, graph_attribute_scaler, target_scaler, num_graph_attributes=6):
    """
    使用训练好的模型对新的图数据进行预测。

    Args:
        model (nn.Module): 已训练的模型。
        graph_file (str): 图的结构文件路径（CSV 格式）。
        parameter_file (str): 图的参数文件路径（CSV 格式）。
        device (torch.device): 计算设备（'cpu' 或 'cuda'）。
        shape_encoder (OneHotEncoder): 形状编码器。
        material_encoder (OneHotEncoder): 材质编码器。
        node_feature_scaler (MinMaxScaler): 节点特征归一化器。
        graph_attribute_scaler (MinMaxScaler): 图属性归一化器。
        target_scaler (MinMaxScaler): 目标归一化器。
        num_graph_attributes (int): 图的属性维度。

    Returns:
        np.ndarray: 反归一化后的预测结果。
    """
    # 创建用于预测的图数据
    graph, batch = create_single_graph_for_prediction(
        graph_file, parameter_file, device, shape_encoder, material_encoder,
        node_feature_scaler, graph_attribute_scaler, num_graph_attributes
    )

    # 确保模型处于评估模式
    model.eval()

    with torch.no_grad():
        # 进行预测
        output = model(graph.x, graph.edge_index, batch, graph.graph_attr)
        output = output.cpu().numpy()

    # 反归一化预测结果
    predictions = target_scaler.inverse_transform(output)

    return predictions


def main():
    # 直接在代码中指定文件路径
    model_path = 'best_model1.pth'  # 训练好的模型路径
    preprocessors_dir = 'preprocessors'  # 预处理器文件所在目录
    graph_file = 'graph_GeO2_solid_circle_2rings_18.0um_2.0um_7.0um.csv'  # 新图结构文件路径
    parameter_file = '1parameter_GeO2_solid_circle_2rings_18.0um_2.0um_7.0um.csv'  # 新图参数文件路径
    output_file = 'predictions.csv'  # 预测结果输出路径

    # 检查文件路径是否存在
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.isfile(graph_file):
        raise FileNotFoundError(f"Graph structure file not found: {graph_file}")
    if not os.path.isfile(parameter_file):
        raise FileNotFoundError(f"Figure parameter file not found: {parameter_file}")

    # 加载预处理器
    shape_encoder, material_encoder, node_feature_scaler, graph_attribute_scaler, target_scaler = load_preprocessors(
        preprocessors_dir)

    # 创建模型实例
    # 请根据训练时的实际参数进行调整
    num_node_features = 4 + shape_encoder.transform([['circle']]).shape[1]  # 4 原始特征 + 形状编码维度
    hidden_dim = 64
    output_dim = 5
    num_graph_attributes = 6

    model = GINModel(
        num_node_features=num_node_features,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_graph_attributes=num_graph_attributes
    ).to(device)

    # 加载训练好的模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # 进行预测
    predictions = predict(
        model,
        graph_file,
        parameter_file,
        device,
        shape_encoder,
        material_encoder,
        node_feature_scaler,
        graph_attribute_scaler,
        target_scaler,
        num_graph_attributes
    )

    # 将预测结果保存到 CSV 文件
    df_predictions = pd.DataFrame(predictions, columns=[f'pred_{i + 1}' for i in range(predictions.shape[1])])
    df_predictions.to_csv(output_file, index=False)
    print(f"The prediction results have been saved to {output_file}")


if __name__ == "__main__":
    main()
