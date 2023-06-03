from torch import nn
from typing import Tuple


def get_head(
    n_classes: int, input_dims: int, layer_sizes: Tuple[int, int, int] = (64, 64)
):
    return ConvHead(n_classes=n_classes, input_dims=input_dims, layer_sizes=layer_sizes)


class ConvHead(nn.Module):
    def __init__(
        self,
        n_classes: int,
        input_dims: int = 384,
        layer_sizes: Tuple[int, int, int] = (64, 64),
    ):
        super().__init__()
        conv_layers = []
        for i in range(len(layer_sizes)):
            in_dim = layer_sizes[i - 1] if i else input_dims
            out_dim = layer_sizes[i]
            conv_layers.extend(
                [
                    nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1),
                    nn.ReLU(),
                ]
            )
        conv_layers.extend(
            [
                nn.Conv2d(
                    in_channels=layer_sizes[-1], out_channels=n_classes, kernel_size=1
                )
            ]
        )
        self.layers = nn.Sequential(*conv_layers)
        self.n_classes = n_classes

    def forward(self, x):
        return self.layers(x)
