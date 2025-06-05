import torch
from torch.nn.utils import parameters_to_vector

def writer_add_params(writer, model, i):
    writer.add_scalar('a/Gradients Norm', parameters_to_vector(p.grad.norm() for p in model.parameters()).norm(), i)
    writer.add_scalar('a/Weights Norm', parameters_to_vector(p.norm() for name, p in model.named_parameters()).norm(), i)
    for name, param in model.named_parameters():
        if 'bias' not in name:
            writer.add_histogram('params/' + name.replace('.', '/'), param, i)
            writer.add_histogram('grad/' + name.replace('.', '/'), param.grad, i)  # note this is a sample from the last mini-batch
