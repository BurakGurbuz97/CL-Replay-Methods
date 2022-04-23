import torch
import torch.nn as nn
from backbone import MammothBackbone, xavier, num_flat_features
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



class View(nn.Module):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return 'View()'

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        out = input.view(-1, 128 *  64)
        return out


class Conv(MammothBackbone):
    """
    Network composed of two hidden layers, each containing 100 ReLU activations.
    Designed for the MNIST dataset.
    """

    def __init__(self,output_size) -> None:

        super(Conv, self).__init__()

        self.input_size = 3
        self.output_size = output_size

        self.conv1 = MaskedConv2d(in_channels=self.input_size, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = MaskedConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = MaskedConv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = MaskedConv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)

        self.fc1 = MaskedLinear(128 * 64, 1024)
        self.classifier= MaskedLinear(1024, output_size)

        self._features = nn.Sequential(
            #First Block
            self.conv1,
            nn.ReLU(),
            self.conv2,
            nn.ReLU(),
            nn.MaxPool2d(2),

            #Second Block
            self.conv3,
            nn.ReLU(),
            self.conv4,
            nn.ReLU(),
            nn.MaxPool2d(2),
            View(),
            self.fc1,
            nn.ReLU()
        )   

        self.net = nn.Sequential(self._features, self.classifier)
        self.reset_parameters()

    def set_masks(self, weight_mask, bias_mask):
        i = 0
        for m in self.modules():
            if isinstance(m,(MaskedLinear, MaskedConv2d)):
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

      

      
class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(MaskedConv2d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride, padding, dilation, groups, bias)
        self.mask_flag = False
        self.bias_flag = bias

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
        if self.mask_flag == True:
            weight = self.weight * self.weight_mask
            if self.bias_flag == True:
                bias = self.bias * self.bias_mask
            else:
                bias = self.bias
            return F.conv2d(x, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)