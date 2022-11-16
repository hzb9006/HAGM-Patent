#!/usr/bin/env python
# coding:utf-8

import pickle
from collections import Counter
import helper.logger as logger
import tqdm
import os
import json


class Vocab(object):
    def __init__(self, config, min_freq=1, special_token=['<PADDING>', '<OOV>'], max_size=None):
        """
        vocabulary class for text classification, initialized from pretrained embedding file
        and update based on minimum frequency and maximum size
        :param config: helper.configure, Configure Object
        :param min_freq: int, the minimum frequency of tokens
        :param special_token: List[Str], e.g. padding and out-of-vocabulary
        :param max_size: int, maximum size of the overall vocabulary
        """
        logger.info('Building Vocabulary....')
        self.corpus_files = {"TRAIN": os.path.join(config.data.data_dir, config.data.train_file),
                             "VAL": os.path.join(config.data.data_dir, config.data.val_file),
                             "TEST": os.path.join(config.data.data_dir, config.data.test_file)}
        counter = Counter() # from collections import Counter，创建一个字典计算数目，键值对，键是str,值是出现的数目，以后计算可以使用这个库
        self.config = config
        # counter for tokens
        self.freqs = {'token': counter.copy(), 'label': counter.copy()}
        # vocab to index
        self.v2i = {'token': dict(), 'label': dict()}
        # index to vocab
        self.i2v = {'token': dict(), 'label': dict()}

        self.min_freq = max(min_freq, 1)
        if not os.path.isdir(self.config.vocabulary.dir):
            os.system('mkdir ' + str(self.config.vocabulary.dir))
        token_dir = os.path.join(self.config.vocabulary.dir, self.config.vocabulary.vocab_dict) # 这是前面使用处理数据集生成的文件
        label_dir = os.path.join(self.config.vocabulary.dir, self.config.vocabulary.label_dict) # 这是前面使用处理数据集生成的文件
        vocab_dir = {'token': token_dir, 'label': label_dir}
        if os.path.isfile(label_dir) and os.path.isfile(token_dir):
            logger.info('Loading Vocabulary from Cached Dictionary...')
            with open(token_dir, 'r',encoding='utf-8') as f_in:
                for i, line in enumerate(f_in):
                    data = line.rstrip().split('\t')
                    # assert len(data) == 2
                    self.v2i['token'][data[0]] = i # 词表到索引
                    self.i2v['token'][i] = data[0] # todo: 等下去看看词表是怎么生成的
            with open(label_dir, 'r') as f_in: # 建立标签的索引，一共有141个
                for i, line in enumerate(f_in):
                    data = line.rstrip().split('\t')
                    assert len(data) == 2
                    self.v2i['label'][data[0]] = i
                    self.i2v['label'][i] = data[0]
            for vocab in self.v2i.keys():
                logger.info('Vocabulary of ' + vocab + ' ' + str(len(self.v2i[vocab])))
        else:
            logger.info('Generating Vocabulary from Corpus...')
            # self._load_pretrained_embedding_vocab() # 加载预训练的词向量文件，考虑分词+中文的，或者不分词,此时注释掉，就是不使用预训练的词向量
            self._count_vocab_from_corpus()
            for vocab in self.freqs.keys():
                logger.info('Vocabulary of ' + vocab + ' ' + str(len(self.freqs[vocab])))

            self._shrink_vocab('token', max_size)
            for s_token in special_token:
                self.freqs['token'][s_token] = self.min_freq # 保证特殊符号不会因为频率太小被删除

            for field in self.freqs.keys():
                temp_vocab_list = list(self.freqs[field].keys())
                for i, k in enumerate(temp_vocab_list):
                    self.v2i[field][k] = i
                    self.i2v[field][i] = k
                logger.info('Vocabulary of ' + field + ' with the size of ' + str(len(self.v2i[field].keys())))
                with open(vocab_dir[field], 'w',encoding='utf-8') as f_out:
                    for k in list(self.v2i[field].keys()):
                        f_out.write(k + '\t' + str(self.freqs[field][k]) + '\n')
                logger.info('Save Vocabulary in ' + vocab_dir[field])
        self.padding_index = self.v2i['token']['<PADDING>']
        self.oov_index = self.v2i['token']['<OOV>']

    def _load_pretrained_embedding_vocab(self):
        """
        initialize counter for word in pre-trained word embedding
        """
        pretrained_file_dir = self.config.embedding.token.pretrained_file # 这里使用的是英文预训练的词向量，我们需要找到中文的，并且随机初始化标签的词向量。。在此之前应该分词
        with open(pretrained_file_dir, 'r', encoding='utf8') as f_in:
            logger.info('Loading vocabulary from pretrained embedding...')
            for line in tqdm.tqdm(f_in):
                data = line.rstrip('\n').split(' ')
                if len(data) == 2:
                    # first line in pretrained embedding
                    continue
                v = data[0]
                self.freqs['token'][v] += self.min_freq + 1

    def _count_vocab_from_corpus(self):
        """
        count the frequency of tokens in the specified corpus
        """
        for corpus in self.corpus_files.keys():
            mode = 'ALL'
            with open(self.corpus_files[corpus], 'r',encoding='utf-8') as f_in:
                logger.info('Loading ' + corpus + ' subset...')
                for line in tqdm.tqdm(f_in):
                    data = json.loads(line.rstrip())
                    self._count_vocab_from_sample(data, mode)

    def _count_vocab_from_sample(self, line_dict, mode='ALL'):
        """
        update the frequency from the current sample
        :param line_dict: Dict{'token': List[Str], 'label': List[Str]}
        """
        for k in self.freqs.keys():
            if mode == 'ALL':
                for t in line_dict[k]:
                    self.freqs[k][t] += 1
            else:
                for t in line_dict['token']:
                    self.freqs['token'][t] += 1

    def _shrink_vocab(self, k, max_size=None):
        """
        shrink the vocabulary
        :param k: Str, field <- 'token', 'label'
        :param max_size: int, the maximum number of vocabulary
        """
        logger.info('Shrinking Vocabulary...')
        tmp_dict = Counter()
        for v in self.freqs[k].keys():
            if self.freqs[k][v] >= self.min_freq:
                tmp_dict[v] = self.freqs[k][v]
        if max_size is not None:
            tmp_list_dict = tmp_dict.most_common(max_size)
            self.freqs[k] = Counter()
            for (t, v) in tmp_list_dict:
                self.freqs[k][t] = v
        logger.info('Shrinking Vocabulary of tokens: ' + str(len(self.freqs[k])))
