import torch.nn as nn


def get_norm_layers(model:nn.Module, norm_name):
    norm_layers = []
    for module in model.modules():
        if isinstance(module, norm_name):
            norm_layers.append(module)
        elif isinstance(module, nn.ModuleList):
            for sub_module in module:
                if isinstance(sub_module, norm_name):
                    norm_layers.append(sub_module)
        elif isinstance(module, nn.Sequential):
            for sub_module in module.children():
                for layer in list(sub_module.modules()):
                    if isinstance(layer, norm_name):
                        layer.requires_grad_(True)
                # norm_layers.append(sub_module)

    return norm_layers


def freeze_weights(model:nn.Module, norm_name):
    for param in model.parameters():
        param.requires_grad = False
    for layer in get_norm_layers(model, norm_name):
        for param in layer.parameters():
            param.requires_grad = True





