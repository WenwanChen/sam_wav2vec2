# train/valid wav to json

import os
import sys
import re
import json
import random
import logging
import glob
from scipy.io import wavfile
from speechbrain.utils.data_utils import get_all_files
from speechbrain.dataio.dataio import read_audio

def split_sets(wav_list, split_ratio):
    # Random shuffle of the list
    random.shuffle(wav_list)
    tot_split = sum(split_ratio)
    tot_snts = len(wav_list)
    data_split = {}
    splits = ["train", "valid"]

    for i, split in enumerate(splits):
        n_snts = int(tot_snts * split_ratio[i] / tot_split)
        data_split[split] = wav_list[0:n_snts]
        # del wav_list[0:n_snts]
    # data_split["test"] = wav_list

    return data_split


def create_json(wav_list, json_file):
    json_dict = {}
    uttid = 0
    for wav_file in wav_list:
        # category = wav_file.split("/")[4]
        signal = read_audio(wav_file)
        duration = signal.shape[0] / 16000
        name = wav_file.split('/')[-2]
        json_dict[str(uttid)] = {
            "wav": wav_file,
            "length": duration,
            "label": name[0],
        }
        uttid += 1

    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)



# # Creating json files for testing and validation data
split_ratio=[85, 15]
wav_list = get_all_files("curated_train_med",match_and=[".wav"])
data_split = split_sets(wav_list, split_ratio)
create_json(data_split["train"], "train.json")
create_json(data_split["valid"], "valid.json")