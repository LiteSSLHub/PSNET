import os
import csv
import json
import logging
import unicodedata

import torch
from pyarrow.json import read_json

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"


def get_logger(name):
    """Initializes multi-GPU-friendly python command line logger."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s')
    logger = logging.getLogger(name)

    return logger


def read_jsonl(fp):
    raw = read_json(fp)
    return WarpJsonObject(raw)


def write_jsonl(fp, data):
    with open(fp, "w") as f:
        f.writelines([json.dumps(it) + "\n" for it in data])

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


class WarpJsonObject:
    """convert pyarrow.Table to python dict"""

    def __init__(self, table):
        self._table = table
        self._feats = table.column_names

        self._start_idx = 0

    def __len__(self):
        return len(self._table)

    def __getitem__(self, index: int):
        return {k: self._table[k][index].as_py() for k in self._feats}

    def __iter__(self):
        self._start_idx = -1
        return self

    def __next__(self):
        self._start_idx += 1

        if self._start_idx == len(self):
            raise StopIteration

        return self[self._start_idx]


def save_checkpoints(filename, ckpt):
    dirname = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filename, "wb") as f:
        torch.save(ckpt, f)


def load_checkpoints(filename, device):
    obj = torch.load(filename, map_location=torch.device(device))
    return obj


def is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def clean_text(text):
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or is_control(char):
            continue
        if is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def parameter_amount(model):
    amount = 0
    for n, p in model.named_parameters():
        amount += p.nelement()
    return amount

def cnn_dataset_to_jsonl(fp, dataset):
    dict_list = []
    for item in dataset:
        dict_list.append({"src": item[2], "tgt": item[3], "labels": item[1]})
    write_jsonl(fp, dict_list)