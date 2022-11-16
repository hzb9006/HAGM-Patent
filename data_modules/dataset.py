#!/usr/bin/env python
# coding: utf-8

from torch.utils.data.dataset import Dataset
import helper.logger as logger
import json
import os


def get_sample_position(corpus_filename, on_memory, corpus_lines, stage):
    """
    position of each sample in the original corpus File or on-memory List
    :param corpus_filename: Str, directory of the corpus file
    :param on_memory: Boolean, True or False
    :param corpus_lines: List[Str] or None, on-memory Data
    :param mode: Str, 'TRAIN'/'DEV'/'TEST', log the corpus
    :return: sample_position -> List[int]
    """
    sample_position = [0] # 能按行读取之后，这是获取每一行句子开始的位置的列表，如sample_position=[0,882,2474]代表第一句话长度为882，第二句话长度为2474-882
    if not on_memory:
        print('Loading files for ' + stage + ' Dataset...')
        with open(corpus_filename, 'r',encoding='utf-8') as f_in: #加载文件
            sample_str = f_in.readline()
            while sample_str:
                sample_position.append(f_in.tell()) # tell() 函数用于判断文件指针当前所处的位置，当使用 open() 函数打开文件时，文件指针的起始位置为 0，表示位于文件的开头处，当使用 read() 函数从文件中读取 3 个字符之后，文件指针同时向后移动了 3 个字符的位置。这就表明，当程序使用文件对象读写数据时，文件指针会自动向后移动：读写了多少个数据，文件指针就自动向后移动多少个位置。
                sample_str = f_in.readline()
            sample_position.pop() # 删除最后一个列表值，因为我们只要知道开始位置，不要知道结束位置
    else:
        assert corpus_lines
        sample_position = range(len(corpus_lines))
    return sample_position


class ClassificationDataset(Dataset):
    def __init__(self, config, vocab, stage='TRAIN', on_memory=True, corpus_lines=None, mode="TRAIN"):
        """
        Dataset for text classification based on torch.utils.data.dataset.Dataset
        :param config: helper.configure, Configure Object
        :param vocab: data_modules.vocab, Vocab Object
        :param stage: Str, 'TRAIN'/'DEV'/'TEST', log the corpus
        :param on_memory: Boolean, True or False
        :param corpus_lines: List[Str] or None, on-memory Data
        :param mode: TRAIN / PREDICT, for loading empty label
        """
        super(ClassificationDataset, self).__init__()
        self.corpus_files = {"TRAIN": os.path.join(config.data.data_dir, config.data.train_file),
                             "VAL": os.path.join(config.data.data_dir, config.data.val_file),
                             "TEST": os.path.join(config.data.data_dir, config.data.test_file)} # 通过配置获取数据集的路径
        self.config = config # 获取配置信息
        self.vocab = vocab
        self.on_memory = on_memory
        self.data = corpus_lines
        self.max_input_length = self.config.text_encoder.max_length
        self.corpus_file = self.corpus_files[stage]
        self.sample_position = get_sample_position(self.corpus_file, self.on_memory, corpus_lines, stage) # 输入文件存储的位置来获取，按行读取的时候，每一个样本开始的位置序号，使用tell函数获取指针位置得到的
        self.corpus_size = len(self.sample_position) # 有几个句子开始的位置就有几句话，所以获取token的size
        self.mode = mode

    def __len__(self):
        """
        get the number of samples
        :return: self.corpus_size -> Int
        """
        return self.corpus_size

    def __getitem__(self, index):
        """
        sample from the overall corpus
        :param index: int, should be smaller in len(corpus)
        :return: sample -> Dict{'token': List[Str], 'label': List[Str], 'token_len': int}
        """
        if index >= self.__len__():
            raise IndexError
        if not self.on_memory:
            position = self.sample_position[index]
            with open(self.corpus_file,encoding='utf-8') as f_in:
                f_in.seek(position)
                sample_str = f_in.readline()
        else:
            sample_str = self.data[index]
        return self._preprocess_sample(sample_str)

    def _preprocess_sample(self, sample_str):
        """
        preprocess each sample with the limitation of maximum length and pad each sample to maximum length
        :param sample_str: Str format of json data, "Dict{'token': List[Str], 'label': List[Str]}"
        :return: sample -> Dict{'token': List[int], 'label': List[int], 'token_len': int}
        """
        raw_sample = json.loads(sample_str)
        sample = {'token': [], 'label': []}
        for k in raw_sample.keys():
            if k == 'token':
                sample[k] = [self.vocab.v2i[k].get(v.lower(), self.vocab.oov_index) for v in raw_sample[k]]
            else:
                sample[k] = []
                for v in raw_sample[k]:
                    if v not in self.vocab.v2i[k].keys():
                        logger.warning('Vocab not in ' + k + ' ' + v)
                    else:
                        sample[k].append(self.vocab.v2i[k][v])
        if not sample['token']:
            sample['token'].append(self.vocab.padding_index)
        if self.mode == 'TRAIN':
            assert sample['label'], 'Label is empty'
        else:
            sample['label'] = [0]
        sample['token_len'] = min(len(sample['token']), self.max_input_length)
        padding = [self.vocab.padding_index for _ in range(0, self.max_input_length - len(sample['token']))]
        sample['token'] += padding
        sample['token'] = sample['token'][:self.max_input_length]
        return sample
