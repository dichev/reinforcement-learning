import torch
from torch.nn.utils import parameters_to_vector

def writer_add_params(writer, model, i):
    model_name = model.__class__.__name__
    writer.add_scalar(f'{model_name}/Gradients Norm', parameters_to_vector(p.grad.norm() for p in model.parameters()).norm(), i)
    writer.add_scalar(f'{model_name}/Weights Norm', parameters_to_vector(p.norm() for name, p in model.named_parameters()).norm(), i)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            writer.add_histogram(f'{model_name}/params/' + name.replace('.', '/'), param, i)
            writer.add_histogram(f'{model_name}/grad/' + name.replace('.', '/'), param.grad, i)  # note this is a sample from the last mini-batch
