import os

image_dir = "./outputs/images/"
model_dir = "./outputs/model/"
data_dir = "./data/"
data_filename = "kdd_cup.npz"
model_filename = "intrusion_dagmm.pt"

compressor_state = 'compressor_state'
estimator_state = 'estimator_state'
compressor_opt_state = 'compressor_opt_state'
estimator_opt_state = 'estimator_opt_state'

train_ratio = 0.8

def setup_dirs():
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)