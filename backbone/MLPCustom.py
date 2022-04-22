import torch
import torch.nn as nn
from backbone import MammothBackbone, xavier, num_flat_features


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

        self.fc1 = nn.Linear(self.input_size, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)

        self._features = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.fc3,
            nn.ReLU(),
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