"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.gcn_net import GCNNet
from nets.mlp_net import MLPNet
from nets.brain_net_cnn import BrainNetCNN
from nets.brainpromptg_net import BrainPromptGNet
from nets.brainpromptc_net import BrainPromptCNet
from nets.braingnn_net import BrainGNNNet
from nets.bnt_net import BNTNet
from nets.contrastpool_net import ContrastPoolNet
from nets.lg_gnn_net import LGGNNNet
from nets.ftspl_net import FTSPLNet


def MLP(net_params, trainset):
    return MLPNet(net_params)

def GCN(net_params, trainset):
    return GCNNet(net_params)


def BrainCNN(net_params, trainset):
    return BrainNetCNN(net_params)


def BrainPromptG(net_params, trainset):
    return BrainPromptGNet(net_params)


def BrainPromptC(net_params, trainset):
    return BrainPromptCNet(net_params)


def gnn_model(MODEL_NAME, net_params, trainset):
    models = {
        'GCN': GCN,
        'MLP': MLP,
        'BrainNetCNN': BrainCNN,
        "BrainPromptG": BrainPromptG,
        "BrainPromptC": BrainPromptC,
        'BrainGNN': lambda p, train: BrainGNNNet(p),
        'BNT': lambda p, train: BNTNet(p),
        'ContrastPool': lambda p, train: ContrastPoolNet(p, train),
        'LG-GNN': lambda p, train: LGGNNNet(p, train),
        'fTSPL': lambda p, train: FTSPLNet(p),
    }
    model = models[MODEL_NAME](net_params, trainset)
    model.name = MODEL_NAME
        
    return model
