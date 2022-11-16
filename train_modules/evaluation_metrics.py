#!/usr/bin/env python
# coding:utf-8

import numpy as np


def _precision_recall_f1(right, predict, total):
    """
    :param right: int, the count of right prediction --真实值为正，预测值也为正，TP
    :param predict: int, the count of prediction--某标签被预测的次数，TP+FP
    :param total: int, the count of labels--真实样本中，某标签应该被预测多少次
    :return: p(precision, Float), r(recall, Float), f(f1_score, Float)--如第一次循环时，对于标签cs: predict=4655，right=3313，total=4239 代表cs被预测了4655次，但是只有3313次是预测正确的，真实样本中有4239属于cs
    """
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict # p=right/predict  per-class precision：某标签被预测正确的次数（该样本属于某标签，并且被预测到了）/某标签被预测的次数（可能包括一些样本不属于该标签，但是被误分为该标签）。
    if total > 0:
        r = float(right) / total # r=right/total  per-class recall：某标签被预测正确了多少次/某标签应该被预测多少次
    if p + r > 0:
        f = p * r * 2 / (p + r) # 计算 micro-F1
    return p, r, f


def evaluate(epoch_predicts, epoch_labels, vocab, threshold=0.5, top_k=None):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    label2id = vocab.v2i['label']
    id2label = vocab.i2v['label']
    epoch_gold_label = list()
    # get id label name of ground truth
    for sample_labels in epoch_labels:
        sample_gold = []
        for label in sample_labels:
            assert label in id2label.keys(), print(label)
            sample_gold.append(id2label[label])
        epoch_gold_label.append(sample_gold) # 根据索引把数字标签转化为文本标签

    epoch_gold = epoch_labels

    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(label2id.keys()))] for _ in range(len(label2id.keys()))] # 有141个值的列表，每个值下面又有141个值
    right_count_list = [0 for _ in range(len(label2id.keys()))] # 正确计数的列表
    gold_count_list = [0 for _ in range(len(label2id.keys()))] # 真实情况下，某个标签应该被预测多少次
    predicted_count_list = [0 for _ in range(len(label2id.keys()))] # 计数，某个标签被预测了多少次

    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        np_sample_predict = np.array(sample_predict, dtype=np.float32)
        sample_predict_descent_idx = np.argsort(-np_sample_predict) #  np.argsort（a）：将a中的元素从小到大排列，提取其在排列前对应的index(索引)输出。此处加了-号，所以是按照从大到小排列，获取索引
        sample_predict_id_list = []
        if top_k is None: # topk是为了了取前面k个，但是由于此处要输出的标签数不固定，是多标签分类问题，所以此处topk=num_label=141
            top_k = len(sample_predict)
        for j in range(top_k):
            if np_sample_predict[sample_predict_descent_idx[j]] > threshold: # 如果预测的概率值大于threshold（0.5），
                sample_predict_id_list.append(sample_predict_descent_idx[j]) # 则对应的label可以作为此句子的标签，把对应label的索引放入sample_predict_id_list

        sample_predict_label_list = [id2label[i] for i in sample_predict_id_list] # 根据上面放入的label的索引查找label的名字放入sample_predict_label_list

        for i in range(len(confusion_count_list)):
            for predict_id in sample_predict_id_list:
                confusion_count_list[i][predict_id] += 1 # todo:此处没懂,会导致某一列都为1，如预测标签为9，则confusion_count_list的第9列全部加1

        # count for the gold and right items
        for gold in sample_gold:
            gold_count_list[gold] += 1 # gold_count_list代表真实标签计数
            for label in sample_predict_id_list:
                if gold == label:
                    right_count_list[gold] += 1 # 如果真是标签id和预测的id一致，则预测正确，right_count_list表示某个标签被预测正确的次数

        # count for the predicted items
        for label in sample_predict_id_list:
            predicted_count_list[label] += 1 # predicted_count_list代表某标签被预测的次数

    precision_dict = dict() #
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0 # 对于wos测试集

    for i, label in id2label.items():
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Macro-F1 要把不同标签的p，r，f进行求和后取平均
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys())) # wos数据集 len(list(precision_dict.keys()))=141
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1}
