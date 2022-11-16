#!/usr/bin/env python
# coding:utf-8

import numpy as np
import torch
import helper.logger as logger
from torch.nn.init import xavier_uniform_, kaiming_uniform_, xavier_normal_, kaiming_normal_, uniform_

INIT_FUNC = {
    'uniform': uniform_,
    'kaiming_uniform': kaiming_uniform_,
    'xavier_uniform': xavier_uniform_,
    'xavier_normal': xavier_normal_,
    'kaiming_normal': kaiming_normal_
}


class EmbeddingLayer(torch.nn.Module):
    def __init__(self,
                 vocab_map,
                 embedding_dim,
                 vocab_name,
                 config,
                 padding_index=None,
                 pretrained_dir=None,
                 model_mode='TRAIN',
                 initial_type='kaiming_uniform',
                 negative_slope=0, mode_fan='fan_in',
                 activation_type='linear',
                 ): # embedding层初始化方式为：'kaiming_uniform'   mode- 可选为 fan_in 或 fan_out, fan_in 使正向传播时，方差一致;fan_out 使反向传播时，方差一致,具体细节以后再了解
        """
        embedding layer
        :param vocab_map: vocab.v2i[filed] -> Dict{Str: Int}
        :param embedding_dim: Int, config.embedding.token.dimension
        :param vocab_name: Str, 'token' or 'label'
        :param config: helper.configure, Configure Object
        :param padding_index: Int, index of padding word
        :param pretrained_dir: Str,  file path for the pretrained embedding file
        :param model_mode: Str, 'TRAIN' or 'EVAL', for initialization
        :param initial_type: Str, initialization type
        :param negative_slope: initialization config
        :param mode_fan: initialization config
        :param activation_type: None
        """
        super(EmbeddingLayer, self).__init__()
        self.dropout = torch.nn.Dropout(p=config['embedding'][vocab_name]['dropout'])
        self.embedding = torch.nn.Embedding(len(vocab_map), embedding_dim, padding_index) # torch.nn.Embedding 随机初始化词向量，词向量值在正态分布N(0,1)中随机取值。

        # initialize lookup table
        assert initial_type in INIT_FUNC
        if initial_type.startswith('kaiming'):
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                    embedding_dim),
                                                        a=negative_slope,
                                                        mode=mode_fan,
                                                        nonlinearity=activation_type)
        elif initial_type.startswith('xavier'):
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                    embedding_dim),
                                                        gain=torch.nn.init.calculate_gain(activation_type))
        else:
            self.lookup_table = INIT_FUNC[initial_type](torch.empty(len(vocab_map),
                                                                    embedding_dim),
                                                        a=-0.25,
                                                        b=0.25) # 等同于torch.nn.init.uniform_（）

        if model_mode == 'TRAIN' and config['embedding'][vocab_name]['type'] == 'pretrain' \
                and pretrained_dir is not None and pretrained_dir != '':
            self.load_pretrained(embedding_dim, vocab_map, vocab_name, pretrained_dir) # 加载预训练词嵌入

        if padding_index is not None:
            self.lookup_table[padding_index] = 0.0 # 根据下标进行索引，用函数表达是index_select()，lookup_table中的第50000行是padding_index对应的随机初始化的嵌入，此处设为0
        self.embedding.weight.data.copy_(self.lookup_table) # 复制我们处理过的（如果预训练嵌入有对应的嵌入，就使用预训练，否则就随机初始化，并且把pad的嵌入改为0）的嵌入到embedding
        self.embedding.weight.requires_grad = True
        del self.lookup_table #删除我们处理的嵌入

    def load_pretrained(self, embedding_dim, vocab_map, vocab_name, pretrained_dir):
        """
        load pretrained file
        :param embedding_dim: Int, configure.embedding.field.dimension
        :param vocab_map: vocab.v2i[field] -> Dict{v:id}
        :param vocab_name: field
        :param pretrained_dir: str, file path
        """
        logger.info('Loading {}-dimension {} embedding from pretrained file: {}'.format(
            embedding_dim, vocab_name, pretrained_dir))
        with open(pretrained_dir, 'r', encoding='utf8') as f_in:
            num_pretrained_vocab = 0 # 统计使用预训练的词嵌入的词的数量
            for line in f_in:
                row = line.rstrip('\n').split(' ')
                if len(row) == 2:
                    assert int(row[1]) == embedding_dim, 'Pretrained dimension %d dismatch the setting %d' \
                                                         % (int(row[1]), embedding_dim)
                    continue
                if row[0] in vocab_map: # 判断预训练的词嵌入对应的词在不在词表里面，如果在的话，就使用预训练的嵌入
                    current_embedding = torch.FloatTensor([float(i) for i in row[1:]])
                    self.lookup_table[vocab_map[row[0]]] = current_embedding # 以预训练的嵌入代替随机初始化的嵌入
                    num_pretrained_vocab += 1
        logger.info('Total vocab size of %s is %d.' % (vocab_name, len(vocab_map)))
        logger.info('Pretrained vocab embedding has %d / %d' % (num_pretrained_vocab, len(vocab_map)))

    def forward(self, vocab_id_list):
        """
        :param vocab_id_list: torch.Tensor, (batch_size, max_length)
        :return: embedding -> torch.FloatTensor, (batch_size, max_length, embedding_dim)
        """
        embedding = self.embedding(vocab_id_list)
        return self.dropout(embedding)
