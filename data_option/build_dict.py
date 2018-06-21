# -*- coding: utf-8 -*-
import config as cfg
import os
import json
from tools.file_option import make_dir_tree


def generate_str_2_label(dict_txt, save_file: str):
    make_dir_tree(os.path.dirname(save_file))
    assert os.path.exists(dict_txt)

    if not save_file.endswith('.json'):
        raise ValueError('save path {:s} should be a json file'.format(save_file))
    char_dict = dict()

    with open(file=dict_txt, mode='r', encoding='utf-8-sig') as txt_f:
        for idx, record in enumerate(txt_f.readlines()):
            record_key = record.strip().split()[0]
            record_value = str(idx)
            char_dict[record_key] = record_value

    with open(save_file, 'w', encoding='utf-8') as json_f:
        json.dump(char_dict, json_f)


def generate_label_2_str(dict_txt, save_file: str):
    make_dir_tree(os.path.dirname(save_file))
    assert os.path.exists(dict_txt)

    if not save_file.endswith('.json'):
        raise ValueError('save path {:s} should be a json file'.format(save_file))
    char_dict = dict()

    with open(file=dict_txt, mode='r', encoding='utf-8-sig') as txt_f:
        for idx, record in enumerate(txt_f.readlines()):
            record_value = record.strip().split()[0]
            record_key = str(idx)
            char_dict[record_key] = record_value

    with open(save_file, 'w', encoding='utf-8') as json_f:
        json.dump(char_dict, json_f)


if __name__ == '__main__':
    pass

    # generate_str_2_label(cfg.PATH_DICT_TXT, cfg.PATH_STR_2_LABEL_DICT)
    # generate_label_2_str(cfg.PATH_DICT_TXT, cfg.PATH_LABEL_2_STR_DICT)

    # with open(cfg.PATH_STR_2_LABEL_DICT, 'r', encoding='utf-8') as json_f:
    #         res1 = json.load(json_f)
    #
    # with open(cfg.PATH_LABEL_2_STR_DICT, 'r', encoding='utf-8') as json_f:
    #         res2 = json.load(json_f)
    # pass
