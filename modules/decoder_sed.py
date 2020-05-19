#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import Tensor, max
from torch.nn import Module, GRU, Linear, Dropout

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['Decoder_sed']


class Decoder_sed(Module):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 nb_classes: int,
                 dropout_p: float) \
            -> None:
        """SED Decoder with no attention.

        :param input_dim: Input features in the decoder.
        :type input_dim: int
        :param output_dim: Output features of the RNN.
        :type output_dim: int
        :param nb_classes: Number of output classes.
        :type nb_classes: int
        :param dropout_p: RNN dropout.
        :type dropout_p: float
        """
        super().__init__()

        self.dropout: Module = Dropout(p=dropout_p)

        self.rnn: Module = GRU(
            input_size=input_dim,
            hidden_size=output_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False)

        self.classifier: Module = Linear(
            in_features=output_dim,
            out_features=nb_classes)


    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the decoder.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output predictions.
        :rtype: torch.Tensor
        """
        h = self.rnn(self.dropout(x))[0]
        return self.classifier(max(h, 1)[0])


# EOF
