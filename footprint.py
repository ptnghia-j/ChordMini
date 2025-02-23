import torch
from modules.models.Transformer.ChordNet import ChordNet

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size_in_bytes(model):
    return sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)

# Instantiate your model as in train.py
model = ChordNet(n_freq=12, n_classes=274, n_group=3,  
                 f_layer=2, f_head=4, 
                 t_layer=2, t_head=4, 
                 d_layer=2, d_head=4, 
                 dropout=0.2)

total_params = count_parameters(model)
size_bytes = model_size_in_bytes(model)
size_mb = size_bytes / (1024 ** 2)

print(f"Total trainable parameters: {total_params}")
print(f"Model size: {size_mb:.2f} MB")