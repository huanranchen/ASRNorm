from torchvision.models import resnet18,resnet50,resnet101
import torch.fx as fx
from torch.fx.node import Argument, Target
from torch.nn.utils.fusion import fuse_conv_bn_eval
from typing import Type, Dict, Any, Tuple, Iterable, Optional, List, cast
from ASRBnrom import ASRNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import replace_pattern
from torch.fx.passes.shape_prop import ShapeProp
import copy
from collections import defaultdict
import torch.utils.mkldnn as th_mkldnn
import operator
import time
import logging
from enum import Enum


def matches_module_pattern(pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]):
    if len(node.args) == 0:
        return False
    nodes: Tuple[Any, fx.Node] = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True

def _parent_name(target : str) -> Tuple[str, str]:
    """
    Splits a qualname into parent path and last atom.
    For example, `foo.bar.baz` -> (`foo.bar`, `baz`)
    """
    *parent, name = target.rsplit('.', 1)
    return parent[0] if parent else '', name

def replace_node_module(node: fx.Node, modules: Dict[str, Any], new_module: torch.nn.Module):
    assert(isinstance(node.target, str))
    parent_name, name = _parent_name(node.target)
    modules[node.target] = new_module
    setattr(modules[parent_name], name, new_module)


def relace_bn_to_asrn(model):
    from torch.fx import symbolic_trace
    traced: torch.fx.GraphModule = symbolic_trace(model)
    traced.graph.print_tabular()
    modules = dict(traced.named_modules())
    new_graph = copy.deepcopy(traced.graph)
    for n in traced.graph.nodes:
        if n.op == "call_module":
            if type(modules[n.target]) == nn.BatchNorm2d:
                replace_node_module(n,modules,ASRNorm(modules[n.target].num_features))
    return torch.fx.GraphModule(traced,new_graph)