import os
import  numpy as np
import torch
import random
import pandas as pd
import re
import glob
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from torch_geometric.data import Data
from configs import Config
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        self.shape_encoder = OneHotEncoder(sparse_output=False)
        self.material_encoder = OneHotEncoder(sparse_output=False)
        self.node_scaler = MinMaxScaler()
        self.radius_scaler = MinMaxScaler()
        self.wavelength_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.edge_scaler = MinMaxScaler()

        self.shape_encoder.fit(np.array(['circle', 'square', 'ellipse']).reshape(-1, 1))
        self.material_encoder.fit(np.array(['SiO2', 'ZBLAN', 'GeO2']).reshape(-1, 1))


processor = DataProcessor()

def create_no_edge_graph(num_nodes):
    edge_index = torch.empty(2, 0, dtype=torch.long)
    edge_weights = torch.empty(0, dtype=torch.float)
    return edge_index, edge_weights


def create_fully_connected_graph(num_nodes):
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weights = torch.ones(edge_index.shape[1])
    return edge_index, edge_weights


def create_circular_graph(num_nodes):
    edge_index = []
    for i in range(num_nodes):

        edge_index.append([i, (i + 1) % num_nodes])
        edge_index.append([(i + 1) % num_nodes, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weights = torch.ones(edge_index.shape[1])
    return edge_index, edge_weights


def create_random_graph(num_nodes, edge_prob=0.2):
    edge_index = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weights = torch.ones(edge_index.shape[1])
    return edge_index, edge_weights


def create_epsilon_neighborhood_graph(coordinates, epsilon=15):
    if not isinstance(coordinates, torch.Tensor):
        coordinates = torch.tensor(coordinates, dtype=torch.float32)

    num_nodes = coordinates.size(0)
    edge_index = []
    edge_weights = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            dist = torch.norm(coordinates[i] - coordinates[j])
            if dist < epsilon:
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_weights.append(dist.item())

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float32)

    return edge_index, edge_weights


def create_edges_from_coordinates(coordinates, k=3):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(coordinates)
    dist_matrix, indices = neigh.kneighbors(coordinates)

    edge_index = []
    edge_weights = []

    for i in range(len(coordinates)):
        for j, dist in zip(indices[i], dist_matrix[i]):
            if i != j:
                edge_index.append([i, j])
                edge_weights.append(dist)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)

    return edge_index, edge_weights


def load_data():
    all_files = glob.glob(Config.data_path)
    all_graphs = []

    collect_data = {
        'numeric_nodes': [],
        'radius': [],
        'wavelengths': [],
        'targets': [],
        'edges': []
    }

    for f in all_files:
        df = pd.read_csv(f)
        node_data = df.iloc[:, :7].dropna()

        collect_data['numeric_nodes'].append(node_data.iloc[:, :6].values.astype(float))

        collect_data['radius'].append(df.iloc[0, 7])

        dynamic = df.iloc[:, [11] + list(range(12, 17))].dropna()
        collect_data['wavelengths'].extend(dynamic.iloc[:, 0].values)
        collect_data['targets'].append(dynamic.iloc[:, 1:6].values)

        coords = node_data.iloc[:, [0, 1]].values
        _, edge_attr = create_edges_from_coordinates(coords)
        collect_data['edges'].append(edge_attr.numpy())

    processor.node_scaler.fit(np.vstack(collect_data['numeric_nodes']))
    processor.radius_scaler.fit(np.array(collect_data['radius']).reshape(-1, 1))
    processor.wavelength_scaler.fit(np.array(collect_data['wavelengths']).reshape(-1, 1))
    processor.target_scaler.fit(np.vstack(collect_data['targets']))
    processor.edge_scaler.fit(np.concatenate(collect_data['edges']).reshape(-1, 1))

    for file_idx, f in enumerate(all_files):
        df = pd.read_csv(f)
        struct_id = int(re.findall(r'\d+', os.path.basename(f))[0])

        node_data = df.iloc[:, :7].dropna()
        coords = node_data.iloc[:, [0, 1]].values


        numeric_features = processor.node_scaler.transform(
            node_data.iloc[:, :6].values.astype(float)
        )

        shapes = node_data.iloc[:, 6].values.reshape(-1, 1)
        shape_features = processor.shape_encoder.transform(shapes)

        node_features = np.hstack([numeric_features, shape_features])

        edge_index, edge_attr = create_edges_from_coordinates(coords)
        edge_attr = processor.edge_scaler.transform(
            edge_attr.numpy().reshape(-1, 1)
        ).flatten()

        dynamic = df.iloc[:, [11] + list(range(12, 17))].dropna()
        wavelengths = processor.wavelength_scaler.transform(
            dynamic.iloc[:, 0].values.reshape(-1, 1)
        )
        targets = processor.target_scaler.transform(
            dynamic.iloc[:, 1:].values
        )

        for i in range(len(wavelengths)):
            radius = processor.radius_scaler.transform([[df.iloc[0, 7]]])[0][0]
            material = processor.material_encoder.transform([[df.iloc[0, 8]]])[0]
            graph_attr = np.concatenate([
                [radius],
                material,
                [wavelengths[i][0]]
            ])

            graph = Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                y=torch.tensor(targets[i], dtype=torch.float32),
                graph_attr=torch.tensor(graph_attr, dtype=torch.float32),
                struct_id=torch.tensor(struct_id, dtype=torch.long),
                file_name=os.path.basename(f)
            )
            all_graphs.append(graph)

    print(f"Total graphs created: {len(all_graphs)}")
    return all_graphs

def split_dataset(graphs):
    train, test = train_test_split(graphs, test_size=0.2, random_state=Config.seed)
    val, test = train_test_split(test, test_size=0.5, random_state=Config.seed)
    return train, val, test
