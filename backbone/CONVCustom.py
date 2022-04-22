import torch
import torch.nn as nn
from backbone import MammothBackbone, xavier, num_flat_features



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

        self.conv1 = nn.Conv2d(in_channels=self.input_size, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)

        self.fc1 = nn.Linear(128 * 64, 1024)
        self.clf = nn.Linear(1024, output_size)

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

        self.classifier = nn.Linear(400, self.output_size)
        self.net = nn.Sequential(self._features, self.classifier)
        self.reset_parameters()
        
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