
import os
import json
import pickle
import pandas as pd
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from torch_geometric.data import Data
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, global_mean_pool
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from threading import Lock

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

# 初始化 Flask 应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # 更改为一个随机的密钥
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

# 初始化 Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # 登录视图的函数名

# 定义 User 类（不使用数据库）
class User(UserMixin):
    def __init__(self, id, username, password_hash):
        self.id = id
        self.username = username
        self.password_hash = password_hash

# 用户数据文件路径
USERS_FILE = 'users.json'
user_lock = Lock()  # 线程锁，防止并发访问

# 加载用户数据
def load_users():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w') as f:
            json.dump([], f)
    with user_lock:
        with open(USERS_FILE, 'r') as f:
            users = json.load(f)
    return users

# 保存用户数据
def save_users(users):
    with user_lock:
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f, indent=4)

# 用户加载回调
@login_manager.user_loader
def load_user(user_id):
    users = load_users()
    for user in users:
        if user['id'] == int(user_id):
            return User(user['id'], user['username'], user['password'])
    return None

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
    required_files = ['shape_encoder.pkl', 'material_encoder.pkl', 'node_feature_scaler.pkl',
                      'graph_attribute_scaler.pkl', 'target_scaler.pkl']
    preprocessors = {}
    for filename in required_files:
        file_path = os.path.join(preprocessors_dir, filename)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"预处理器文件未找到: {file_path}")
        with open(file_path, 'rb') as f:
            preprocessors[filename.split('.')[0]] = pickle.load(f)
    return (preprocessors['shape_encoder'],
            preprocessors['material_encoder'],
            preprocessors['node_feature_scaler'],
            preprocessors['graph_attribute_scaler'],
            preprocessors['target_scaler'])

# 创建单个图数据用于预测
def create_single_graph_for_prediction(graph_file, parameter_file, device, shape_encoder, material_encoder,
                                       node_feature_scaler, graph_attribute_scaler, num_graph_attributes=6):
    # 检查文件是否存在
    if not os.path.isfile(graph_file):
        raise FileNotFoundError(f"图结构文件未找到: {graph_file}")
    if not os.path.isfile(parameter_file):
        raise FileNotFoundError(f"图参数文件未找到: {parameter_file}")

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

# 加载模型和预处理器
try:
    shape_encoder, material_encoder, node_feature_scaler, graph_attribute_scaler, target_scaler = load_preprocessors(
        'preprocessors')
    print("Preprocessor loaded successfully.")
except Exception as e:
    print(f"Error loading preprocessor: {e}")

# 创建模型实例
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
model_path = 'best_model1.pth'  # 训练好的模型路径
if os.path.isfile(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print("Model weights loaded successfully。")
    except Exception as e:
        print(f"Error loading model weights: {e}")
else:
    print(f"Model file not found: {model_path}")

# 定义路由

@app.route('/')
# @login_required  # 需要登录才能访问
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
# @login_required  # 需要登录才能访问
def make_prediction():
    if 'graph_file' not in request.files or 'parameter_file' not in request.files:
        return jsonify({'error': 'Please upload the graph structure file and graph parameter file.'}), 400

    graph_file = request.files['graph_file']
    parameter_file = request.files['parameter_file']

    if graph_file.filename == '' or parameter_file.filename == '':
        return jsonify({'error': 'Please select a file to upload.'}), 400

    if not allowed_file(graph_file.filename) or not allowed_file(parameter_file.filename):
        return jsonify({'error': 'Only CSV files are allowed to be uploaded.'}), 400

    try:
        # 使用 secure_filename 处理文件名
        graph_filename = secure_filename(graph_file.filename)
        parameter_filename = secure_filename(parameter_file.filename)

        # 保存上传的文件到临时位置
        graph_path = os.path.join('temp', graph_filename)
        parameter_path = os.path.join('temp', parameter_filename)
        os.makedirs('temp', exist_ok=True)
        graph_file.save(graph_path)
        parameter_file.save(parameter_path)

        # 进行预测
        predictions = predict(
            model,
            graph_path,
            parameter_path,
            device,
            shape_encoder,
            material_encoder,
            node_feature_scaler,
            graph_attribute_scaler,
            target_scaler,
            num_graph_attributes
        )

        # 删除临时文件
        os.remove(graph_path)
        os.remove(parameter_path)

        # 返回预测结果
        return jsonify({'predictions': predictions.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 用户注册路由
@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username').strip()
        password = request.form.get('password').strip()
        if not username or not password:
            flash('用户名和密码不能为空。', 'error')
            return redirect(url_for('register'))

        users = load_users()
        for user in users:
            if user['username'] == username:
                flash('用户名已存在。', 'error')
                return redirect(url_for('register'))

        new_id = users[-1]['id'] + 1 if users else 1
        password_hash = generate_password_hash(password, method='sha256')
        new_user = {
            'id': new_id,
            'username': username,
            'password': password_hash
        }
        users.append(new_user)
        save_users(users)
        flash('注册成功，请登录。', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

# 用户登录路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form.get('username').strip()
        password = request.form.get('password').strip()
        if not username or not password:
            flash('用户名和密码不能为空。', 'error')
            return redirect(url_for('login'))

        users = load_users()
        user_data = next((user for user in users if user['username'] == username), None)
        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_data['id'], user_data['username'], user_data['password'])
            login_user(user)
            # flash('登录成功。', 'success')
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('无效的用户名或密码。', 'error')
            return redirect(url_for('login'))
    return render_template('login.html')

# 用户登出路由
@app.route('/logout')
@login_required
def logout():
    logout_user()
    # flash('已登出。', 'success')
    return redirect(url_for('login'))

# 添加系统退出路由（仅管理员可用）
def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/shutdown', methods=['POST'])
@login_required
def shutdown():
    if current_user.username != 'admin':
        return jsonify({'error': '无权限执行此操作。'}), 403
    shutdown_server()
    return '服务器正在关闭...', 200

# 限制上传文件类型
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 启动 Flask 应用
if __name__ == "__main__":
    # 更改端口为5001
    app.run(host='0.0.0.0', port=5001, debug=True)
