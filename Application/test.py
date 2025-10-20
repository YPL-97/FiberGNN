import pickle
import os

preprocessors_dir = 'preprocessors'
shape_encoder_path = os.path.join(preprocessors_dir, 'shape_encoder.pkl')

with open(shape_encoder_path, 'rb') as f:
    shape_encoder = pickle.load(f)

print("Shape Encoder loaded successfully:", shape_encoder)
