import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from backbone import MammothBackbone, xavier, num_flat_features
import numpy as np


class MLP(MammothBackbone):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Instantiates the layers of the network.
        :param input_size: the size of the input data
        :param output_size: the size of the output
        """
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.fc1 = MaskedLinear(self.input_size, 400)
        self.fc2 = MaskedLinear(400, 400)
        self.fc3 = MaskedLinear(400, 400)

        self._features = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3,
            nn.ReLU(),
        )
        self.classifier = MaskedLinear(400, self.output_size)
        self.net = nn.Sequential(self._features, self.classifier)
        self.reset_parameters()

    def set_masks(self, weight_mask, bias_mask):
        i = 0
        for m in self.modules():
            if isinstance(m,(MaskedLinear)):
                m.set_mask(weight_mask[i],bias_mask[i])
                i = i + 1
        
    def reset_parameters(self) -> None:
        """
        Calls the Xavier parameter initialization function.
        """
        self.net.apply(xavier)

    def forward(self, x: torch.Tensor, returnt='out') -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (output_size)
        """
        x = x.view(-1, num_flat_features(x))

        feats = self._features(x)
        
        if returnt == 'features':
            return feats

        out = self.classifier(feats)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return (out, feats)
        
        raise NotImplementedError("Unknown return type")


def to_var(x, requires_grad = False, volatile = False):
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if torch.cuda.is_available():
        x = x.to(torch.device("cuda"))
    return Variable(x, requires_grad = requires_grad, volatile = volatile)

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
        self.bias_flag = bias
        self.sparse_grads = True
        
    def set_mask(self, weight_mask, bias_mask):
        self.weight_mask = to_var(weight_mask, requires_grad=False)
        self.weight.data = self.weight.data * self.weight_mask.data
        if self.bias_flag == True:
            self.bias_mask = to_var(bias_mask, requires_grad=False)
            self.bias.data = self.bias.data * self.bias_mask.data
        self.mask_flag = True

    def get_mask(self):
        return self.weight_mask, self.bias_mask

    def forward(self, x):
        if self.mask_flag == True and self.sparse_grads:
            weight = self.weight * self.weight_mask
            if self.bias_flag == True:
                bias = self.bias * self.bias_mask
            else:
                bias = self.bias
            return F.linear(x, weight, bias)
        else:
            return F.linear(x, self.weight, self.bias)