#!/usr/bin/env python
# coding:utf-8

import torch
from torch import nn
import torch.nn.functional as F

# todo:再次学习一下lstm和GRU，并且学习使用
class GRU(nn.Module):
    def __init__(self,
                 layers,
                 input_dim,
                 output_dim,
                 bias=True,
                 batch_first=False,
                 dropout=0.0,
                 bidirectional=True):
        """
        GRU module
        :param layers: int, the number of layers, config.text_encoder.RNN.num_layers
        :param input_dim: int, config.embedding.token.dimension
        :param output_dim: int, config.text_encoder.RNN.hidden_dimension
        :param bias: None
        :param batch_first: True
        :param dropout: p = dropout, config.text_encoder.RNN.dropout
        :param bidirectional: Boolean , default True, config.text_encoder.RNN.bidirectional
        """
        super(GRU, self).__init__()
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_layers = layers
        self.gru = torch.nn.GRU(input_size=input_dim,
                                hidden_size=output_dim,
                                num_layers=layers,
                                batch_first=batch_first,
                                bias=bias,
                                bidirectional=bidirectional,
                                dropout=dropout)

    def forward(self, inputs, seq_len=None, init_state=None, ori_state=False):
        """
        :param inputs: torch.FloatTensor, (batch, max_length, embedding_dim)
        :param seq_len: torch.LongTensor, (batch, max_length)
        :param init_state: None
        :param ori_state: False
        :return: padding_out -> (batch, max_length, 2 * hidden_dimension),
        """
        if seq_len is not None:
            seq_len = seq_len.int()
            sorted_seq_len, indices = torch.sort(seq_len, descending=True)
            if self.batch_first:
                sorted_inputs = inputs[indices]
            else:
                sorted_inputs = inputs[:, indices]
            packed_inputs = torch.nn.utils.rnn.pack_padded_sequence(
                sorted_inputs,
                sorted_seq_len,
                batch_first=self.batch_first,
            )

            outputs, states = self.gru(packed_inputs, init_state)
        if ori_state:
            return outputs, states
        if self.bidirectional:
            last_layer_hidden_state = states[2 * (self.num_layers - 1):]
            last_layer_hidden_state = torch.cat((last_layer_hidden_state[0], last_layer_hidden_state[1]), 1)
        else:
            last_layer_hidden_state = states[self.num_layers - 1]
            last_layer_hidden_state = last_layer_hidden_state[0]

        _, reversed_indices = torch.sort(indices, descending=False)
        last_layer_hidden_state = last_layer_hidden_state[reversed_indices]
        padding_out, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs,
                                                                batch_first=self.batch_first)
        if self.batch_first:
            padding_out = padding_out[reversed_indices]
        else:
            padding_out = padding_out[:, reversed_indices]
        return padding_out, last_layer_hidden_state


class TextEncoder(nn.Module):
    def __init__(self, config):
        """
        TextRCNN
        :param config: helper.configure, Configure Object
        """
        super(TextEncoder, self).__init__()
        self.config = config
        self.rnn = GRU(
            layers=config.text_encoder.RNN.num_layers,
            input_dim=config.embedding.token.dimension,
            output_dim=config.text_encoder.RNN.hidden_dimension,
            batch_first=True,
            bidirectional=config.text_encoder.RNN.bidirectional
        ) # 使用bilstm 进行文本的编码 batch_first如果为True，则输入和输出的tensors的维度为(batch,seq,feature)而不是(seq,batch,feature)。注意：此标识只对output有效，对hidden state 和 cell state无效，默认为False
        hidden_dimension = config.text_encoder.RNN.hidden_dimension
        if config.text_encoder.RNN.bidirectional:
            hidden_dimension *= 2
        self.kernel_sizes = config.text_encoder.CNN.kernel_size
        self.convs = torch.nn.ModuleList() #加载模块列表
        for kernel_size in self.kernel_sizes: # 自然语言处理中一个句子序列，一维的，所以使用Conv1d
            self.convs.append(torch.nn.Conv1d(
                hidden_dimension,
                config.text_encoder.CNN.num_kernel,
                kernel_size,
                padding=kernel_size // 2
                )
            )
        self.top_k = config.text_encoder.topK_max_pooling
        self.rnn_dropout = torch.nn.Dropout(p=config.text_encoder.RNN.dropout)

    def forward(self, inputs, seq_lens):
        """
        :param inputs: torch.FloatTensor, embedding, (batch, max_len, embedding_dim)
        :param seq_lens: torch.LongTensor, (batch, max_len)
        :return:
        """
        text_output, _ = self.rnn(inputs, seq_lens)
        text_output = self.rnn_dropout(text_output)
        text_output = text_output.transpose(1, 2)
        topk_text_outputs = []
        for _, conv in enumerate(self.convs):
            convolution = F.relu(conv(text_output))
            topk_text = torch.topk(convolution, self.top_k)[0].view(text_output.size(0), -1)
            topk_text = topk_text.unsqueeze(1)
            topk_text_outputs.append(topk_text)
        return topk_text_outputs
