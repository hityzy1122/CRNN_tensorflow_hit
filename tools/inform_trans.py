import config as cfg
import json
import os
import tensorflow as tf
import numpy as np


def trans_str_2_label(string:str):
    assert os.path.exists(cfg.PATH_STR_2_LABEL_DICT)

    with open(cfg.PATH_STR_2_LABEL_DICT, 'r', encoding='utf-8') as json_f:
        str_2_label_dict = json.load(json_f)
    return [int(str_2_label_dict[val]) for val in string]


def trans_label_2_str(label):

    assert os.path.exists(cfg.PATH_LABEL_2_STR_DICT)

    with open(cfg.PATH_LABEL_2_STR_DICT, 'r', encoding='utf-8') as json_f:
        label_2_str_dict = json.load(json_f)
    return label_2_str_dict[str(label)]


def sparse_tensor_to_str(sparse_tensor:tf.SparseTensor):
    indices = sparse_tensor.indices
    values = sparse_tensor.values
    dense_shape = sparse_tensor.dense_shape

    number_lists = np.zeros(dense_shape, dtype=values.dtype)
    str_lists = []
    res = []

    for i, index in enumerate(indices):
        number_lists[index[0], index[1]] = values[i]
    for number_list in number_lists:
        str_lists.append([trans_label_2_str(val) for val in number_list])
    for str_list in str_lists:
        res.append(''.join(c for c in str_list if c != '*'))

    return res


if __name__ == '__main__':
    a = trans_str_2_label('*0123456789X你好呀')

    print(a)

    b = trans_label_2_str([0,1,2,3,4,5,6,7,8,9])

    print(b)