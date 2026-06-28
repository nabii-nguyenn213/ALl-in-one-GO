import torch 
import numpy as np 
import torch.nn as nn 

activations = {
    "linear" : nn.Identity(), 
    "relu"   : nn.ReLU(), 
    "leakyrelu": nn.LeakyReLU(), 
    "tanh"   : nn.Tanh(), 
    "sigmoid": nn.Sigmoid(), 
    "softmax-1": nn.Softmax(dim=-1), 
    "softmax0": nn.Softmax(dim=0), 
    "softmax1": nn.Softmax(dim=1), 
    "softmax2": nn.Softmax(dim=2), 
}

class MLP(nn.Module): 
    def __init__(self, layer_dims, hidden_act="ReLU", output_act="Linear"): 
        super().__init__()
        layers = []
        for i in range(len(layer_dims)-1): 
            act = hidden_act if i+2!=len(layer_dims) else output_act
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1], bias=True))
            layers.append(activations[act.lower()])
        self.mlp = nn.Sequential(*layers)

    def forward(self, x): 
        return self.mlp(x)

class ActorNetwork(nn.Module): 
    pass 

class QCriticNetwork(nn.Module): 
    
    def __init__(self, obs_dim, act_dim, hidden_size=[64, 64], 
                 hidden_act="relu", output_act="linear"): 
        super().__init__()
        layer_dims = [obs_dim+act_dim, *hidden_size, 1]
        self.criticnet = MLP(layer_dims=layer_dims, hidden_act=hidden_act, output_act=output_act)

    def forward(self, obs, act): 
        if act.ndim==1: act = torch.unsqueeze(act, dim=1)
        x = torch.cat([obs, act], dim = 1)
        return self.criticnet(x)

class VCriticNetwork(nn.Module): 
    
    def __init__(self, obs_dim, hidden_size=[64, 64], hidden_act="relu", output_act="linear"): 
        super().__init__()
        layer_dims = [obs_dim, *hidden_size, 1]
        self.vnet = MLP(layer_dims, hidden_act, output_act)

    def forward(self, obs): 
        return self.vnet(obs)
