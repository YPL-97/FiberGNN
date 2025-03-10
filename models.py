import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGEConv,global_mean_pool
from configs import Config


class GCNModel(nn.Module):
    def __init__(self, num_node_features, num_graph_attributes, output_dim, hidden_dim=256):
        super().__init__()
        self.num_graph_attributes = num_graph_attributes
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim + num_graph_attributes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        graph_attr = data.graph_attr.view(-1, self.num_graph_attributes)
        x = torch.cat((x, graph_attr), dim=1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GATModel(nn.Module):
    def __init__(self, num_node_features, num_graph_attributes, output_dim, hidden_dim=256):
        super().__init__()
        self.num_graph_attributes = num_graph_attributes
        self.attn1 = GATConv(num_node_features, hidden_dim)
        self.attn2 = GATConv(hidden_dim, hidden_dim)
        self.attn3 = GATConv(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim + num_graph_attributes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.attn1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.attn2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.attn3(x, edge_index)
        x = self.norm3(x)
        x = F.relu(x)
        x = global_mean_pool(x, batch)

        graph_attr = data.graph_attr.view(-1, self.num_graph_attributes)

        # 现在可以拼接 x 和 graph_attr
        x = torch.cat((x, graph_attr), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GINModel(nn.Module):
    def __init__(self, num_node_features, num_graph_attributes, output_dim, hidden_dim=256):
        super().__init__()
        self.num_graph_attributes = num_graph_attributes
        self.conv1 = GINConv(
            nn.Sequential(nn.Linear(num_node_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim + num_graph_attributes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        x = F.relu(x)
        x = global_mean_pool(x, batch)

        graph_attr = data.graph_attr.view(-1, self.num_graph_attributes)
        # 现在可以拼接 x 和 graph_attr
        x = torch.cat((x, graph_attr), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class GraphSAGEModel(nn.Module):
    def __init__(self, num_node_features, num_graph_attributes, output_dim, hidden_dim=256):
        super().__init__()
        self.num_graph_attributes = num_graph_attributes
        self.conv1 = SAGEConv(num_node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim + num_graph_attributes, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)

        graph_attr = data.graph_attr.view(-1, self.num_graph_attributes)

        x = torch.cat((x, graph_attr), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x