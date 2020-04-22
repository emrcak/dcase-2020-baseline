#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import cat, Tensor
from torch.nn import Module, GRU, Dropout, Conv2d, MaxPool2d, LeakyReLU
from typing import List

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['Encoder']


class Encoder(Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_conv_layers: int,
                 output_dim: int,
                 dropout_p: float) \
            -> None:
        """Encoder module.

        :param input_dim: Input dimensionality.
        :type input_dim: int
        :param hidden_dim: Hidden dimensionality.
        :type hidden_dim: int
        :param output_dim: Output dimensionality.
        :type output_dim: int
        :param dropout_p: Dropout.
        :type dropout_p: float
        """
        super(Encoder, self).__init__()

        self.input_dim: int = input_dim
        self.hidden_dim: int = hidden_dim
        self.output_dim: int = output_dim
        self.num_conv_layers: int = num_conv_layers
        self.rnn_input_dim: int = int(self.input_dim / (2 ** self.num_conv_layers) * self.hidden_dim) if \
            self.num_conv_layers > 0 else self.input_dim

        self.dropout: Module = Dropout(p=dropout_p)

        cnn_common_args = {
            'kernel_size': 3,
            'stride': 1,
            'padding': 1,
            'dilation': 1,
            'bias': True}


        rnn_common_args = {
            'num_layers': 1,
            'bias': True,
            'batch_first': True,
            'bidirectional': True}

        if self.num_conv_layers >= 1:
            self.cnn_1: Module = Conv2d(
                in_channels=1,
                out_channels=self.hidden_dim,
                **cnn_common_args)
            self.leaky_relu = LeakyReLU()
            self.pool: Module = MaxPool2d(kernel_size=2)

        if self.num_conv_layers > 1:
            self.cnn_extra: Module = Conv2d(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                **cnn_common_args)

        self.gru_1: Module = GRU(
            input_size=self.rnn_input_dim,
            hidden_size=self.hidden_dim,
            **rnn_common_args)

        self.gru_2: Module = GRU(
            input_size=self.hidden_dim*2,
            hidden_size=self.hidden_dim,
            **rnn_common_args)

        self.gru_3: Module = GRU(
            input_size=self.hidden_dim*2,
            hidden_size=self.output_dim,
            **rnn_common_args)

    def _cnn_pass(self,
                input: Tensor) \
            -> Tensor:

        c = self.cnn_1(input.unsqueeze(1))
        c = self.leaky_relu(c)
        c = self.pool(c)
        for k in range(1, self.num_conv_layers):
            c = self.cnn_extra(c)
            c = self.leaky_relu(c)
            c = self.pool(c)

        b_size, num_channels, t_steps, num_feats = c.size()
        c = c.permute(0, 2, 3, 1)
        c = c.reshape(b_size, t_steps, -1)
        return c


    def _l_pass(self,
                layer: Module,
                layer_input: Tensor) \
            -> Tensor:
        """Does the forward passing for a GRU layer.

        :param layer: GRU layer for forward passing.
        :type layer: torch.nn.Module
        :param layer_input: Input to the GRU layer.
        :type layer_input: torch.Tensor
        :return: Output of the GRU layer.
        :rtype: torch.Tensor
        """
        b_size, t_steps, _ = layer_input.size()
        h = layer(layer_input)[0].view(b_size, t_steps, 2, -1)
        return self.dropout(cat([h[:, :, 0, :], h[:, :, 1, :]], dim=-1))

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the encoder.

        :param x: Input to the encoder.
        :type x: torch.Tensor
        :return: Output of the encoder.
        :rtype: torch.Tensor
        """

        if self.num_conv_layers > 0:
            c = self._cnn_pass(x)
        else:
            c = x
        h = self._l_pass(self.gru_1, c)

        for a_layer in [self.gru_2, self.gru_3]:
            h_ = self._l_pass(a_layer, h)
            h = h + h_ if h.size()[-1] == h_.size()[-1] else h_

        return h

# EOF
