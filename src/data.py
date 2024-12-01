import os
from functools import partial

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformers import BertTokenizer

from utils import (
    read_jsonl,
    get_logger
)

log = get_logger(__name__)

dataset_info = {
    "sum": {"text_max_len": 512, "max_text_ntokens_per_sent": 200, "min_text_ntokens_per_sent": 5, "label_list": [None],
            "output_mode": 'classification'},
}


# Returns the information of the dataset, including max sequence length, etc. 
# Currently, only the information of the cnn_dailymail dataset is used.
def get_dataset_info(name):
    return dataset_info[name]


# The example classes used for the three classification tasks: agnews, yahoo, dbpedia.
class GlueData(object):
    def __init__(self, text_a=None, text_b=None, label=None):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


# Convert the examples of the three classification tasks: agnews, yahoo, dbpedia into features.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        # Gradually truncate the longer one of the two sentences until its length is less than or equal to max_length
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # The reason for -3 is that it is necessary to add 1 [CLS] and 2 [SEP] tokens.
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        # seq_length = len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label = label_map[example.label]
        else:
            label = float(example.label)

        input_ids = torch.tensor(input_ids)
        input_mask = torch.tensor(input_mask)
        segment_ids = torch.tensor(segment_ids)
        label = torch.tensor(label)

        features.append((input_ids, input_mask, segment_ids, label))
    return features


# The DataModule used for the extractive summarization task, used to generate the 
# dataloader needed for the training and evaluation phases.
class ExtractiveSummaryDataModule:
    def __init__(self, args):
        self.train_filename = os.path.join("./data", args.cnndm_dataset_name, "train.jsonl")
        self.val_filename = os.path.join("./data", args.cnndm_dataset_name, "val.jsonl")
        self.test_filename = os.path.join("./data", args.cnndm_dataset_name, "test.jsonl")

        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        self.test_batch_size = args.test_batch_size

        self.num_workers = args.num_workers

        self.tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name_or_path)

        dataset_info = get_dataset_info("sum")
        self.text_max_len = dataset_info["text_max_len"]
        self.max_text_ntokens_per_sent = dataset_info["max_text_ntokens_per_sent"]
        self.min_text_ntokens_per_sent = dataset_info["min_text_ntokens_per_sent"]

        self.num_labels=len(dataset_info["label_list"])

        self.collator = partial(collator,
                                cls_id=self.tokenizer.cls_token_id,
                                sep_id=self.tokenizer.sep_token_id,
                                pad_id=self.tokenizer.pad_token_id,
                                text_max_len=self.text_max_len)

    def prepare(self, args):
        text_args = {
            "text_max_len": self.text_max_len,
            "max_text_ntokens_per_sent": self.max_text_ntokens_per_sent,
            "min_text_ntokens_per_sent": self.min_text_ntokens_per_sent,
        }
        self.train_dataset = NewsDataset(self.train_filename, self.tokenizer, **text_args)
        self.val_dataset = NewsDataset(self.val_filename, self.tokenizer, **text_args)
        self.test_dataset = NewsDataset(self.test_filename, self.tokenizer, **text_args)

        assert args.supervised_size <= len(
            self.train_dataset), f"The total size of the dataset is {len(self.train_dataset)}!"
        log.info(
            f"Split train dataset: {args.supervised_size} for supervised; data {len(self.train_dataset)} for unsupervised data.")
        supervised_size, remain_size = args.supervised_size, len(self.train_dataset) - args.supervised_size
        self.train_dataset, self.unsupervised_train_dataset = \
            random_split(self.train_dataset, [supervised_size, remain_size])
        # cnn_dataset_to_jsonl("./data/train_100.jsonl", self.train_dataset)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def unsupervised_dataloader(self):
        assert getattr(self,
                       "unsupervised_train_dataset") is not None, "Unsupervised training dataset is not properly loaded!"

        return DataLoader(
            dataset=self.unsupervised_train_dataset + self.train_dataset,
            batch_size=self.train_batch_size * 8,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )


# The DataModule used for the three classification tasks: agnews, yahoo, dbpedia, 
# used to generate the dataloader needed for the training and evaluation phases.
class SelfUdaDataModule:
    def __init__(self, args):
        self.dataset_name = os.path.join("./data", args.glue_dataset_name, "train.jsonl")

        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        self.test_batch_size = args.test_batch_size

        self.num_workers = args.num_workers
        self.tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name_or_path)
        self.task_name = args.task_name

        self.collator = gluecollator

        train_filename = os.path.join("./data", args.glue_dataset_name, "train_labeled_" + str(args.supervised_size) + ".jsonl")
        unlabeled_filename = os.path.join("./data", args.glue_dataset_name, "train_unlabeled.jsonl")
        val_filename = os.path.join("./data", args.glue_dataset_name, "val.jsonl")
        test_filename = os.path.join("./data", args.glue_dataset_name, "test.jsonl")

        self.train_labeled_dataset = SelfUdaDataset(data_filename=train_filename,
                                                    tokenizer=self.tokenizer,
                                                    max_seq_len=256)
        self.train_unlabeled_dataset = SelfUdaDataset(data_filename=unlabeled_filename,
                                                      tokenizer=self.tokenizer,
                                                      max_seq_len=256, )
        self.val_dataset = SelfUdaDataset(data_filename=val_filename,
                                          tokenizer=self.tokenizer,
                                          max_seq_len=256)
        self.test_dataset = SelfUdaDataset(data_filename=test_filename,
                                           tokenizer=self.tokenizer,
                                           max_seq_len=256)
        self.num_labels = len(self.train_labeled_dataset) // args.supervised_size
        log.info("#Labeled: {}, Unlabeled {}, Val {}, Test {}".format(len(
            self.train_labeled_dataset), len(self.train_unlabeled_dataset), len(self.val_dataset),
            len(self.test_dataset)))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_labeled_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def unsupervised_dataloader(self):
        return DataLoader(
            dataset=self.train_unlabeled_dataset,
            batch_size=self.train_batch_size * 4,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            collate_fn=self.collator,
        )


# The dataset class used for the extractive summarization task
class NewsDataset(Dataset):
    def __init__(self, data_filename, tokenizer, text_max_len, max_text_ntokens_per_sent, min_text_ntokens_per_sent):
        self.data = read_jsonl(data_filename)

        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        self.max_text_ntokens_per_sent = max_text_ntokens_per_sent
        self.min_text_ntokens_per_sent = min_text_ntokens_per_sent

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ex = self.data[index]
        text, summary, labels = ex["src"], ex["tgt"], ex["labels"]

        add_special_token_text = list(map(lambda sent: self.tokenizer.cls_token + sent, text))
        subtokens = list(map(self.tokenizer.tokenize, add_special_token_text))

        mask_idxs = [i for i, t in enumerate(subtokens) if len(t) > self.min_text_ntokens_per_sent]

        subtokens = [subtokens[idx] for idx in mask_idxs]
        text = [text[idx] for idx in mask_idxs]
        labels = [labels[idx] for idx in mask_idxs]

        sent_ids = list(map(self.tokenizer.convert_tokens_to_ids, subtokens))
        sent_ids = [ids[:self.max_text_ntokens_per_sent - 1] + [self.tokenizer.sep_token_id] for ids in sent_ids]

        text_ids = list()
        for i in range(len(sent_ids)):
            if len(text_ids) + len(sent_ids[i]) <= self.text_max_len:
                text_ids.extend(sent_ids[i])
            else:
                remain_len = self.text_max_len - len(text_ids)
                if remain_len > self.min_text_ntokens_per_sent:
                    text_ids.extend(sent_ids[i][:remain_len - 1] + [self.tokenizer.sep_token_id])
                break

        text_ids = torch.tensor(text_ids)

        sent_num = (text_ids == self.tokenizer.cls_token_id).sum()

        sent_ids = sent_ids[:sent_num]
        labels = labels[:sent_num]

        return text_ids, labels, text, summary


# The dataset class used for the three classification tasks: agnews, yahoo, dbpedia.
class SelfUdaDataset(Dataset):
    def __init__(self, data_filename, tokenizer, max_seq_len):
        lines = read_jsonl(data_filename)
        selfudadata = []
        self.n_labels = 0
        for (i, line) in enumerate(lines):
            selfudadata.append(
                GlueData(text_a=line["text"], label=line["label"]))
            if line["label"] > self.n_labels:
                self.n_labels = line["label"]
        self.data = convert_examples_to_features(selfudadata, range(self.n_labels + 1), max_seq_len, tokenizer,
                                                 "classification")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


# =========================================================COLLATOR=============================================================


# The custom batch class, convenient for performing the .to(device) operation 
# when retrieving data for each batch using the dataloader.
class Batch:
    def __init__(
            self, input_ids=None, attn_mask=None, cls_mask=None, sep_mask=None, seg=None, labels=None, texts=None,
            summaries=None):
        self.input_ids = input_ids
        self.attn_mask = attn_mask
        self.cls_mask = cls_mask
        self.sep_mask = sep_mask
        self.seg = seg
        self.labels = labels
        self.texts = texts
        self.summaries = summaries

    def to(self, device):
        self.input_ids = self.input_ids.to(device) if self.input_ids is not None else None
        self.attn_mask = self.attn_mask.to(device) if self.attn_mask is not None else None
        self.cls_mask = self.cls_mask.to(device) if self.cls_mask is not None else None
        self.sep_mask = self.sep_mask.to(device) if self.sep_mask is not None else None
        self.seg = self.seg.to(device) if self.seg is not None else None
        self.labels = self.labels.to(device) if self.labels is not None else None

        return self

    def __len__(self):
        return self.input_ids.size(0)


def pad_1d(x, pad_len, pad_id):
    xlen = x.size(0)
    if xlen < pad_len:
        new_x = x.new_empty([pad_len], dtype=x.dtype).fill_(pad_id)
        new_x[:xlen] = x
        x = new_x
    elif xlen > pad_len:
        end_id = x[-1]
        x = x[:pad_len]
        x[-1] = end_id
    return x.int()


def pad_2d(x, pad_len, pad_id):
    x = x + 1
    xlen, xdim = x.size()
    if xlen < pad_len:
        new_x = x.new_zeros([pad_len, xdim], dtype=x.dtype).fill_(pad_id)
        new_x[:xlen, :] = x
        x = new_x
    return x.int()


# The collator used by the dataloader for the extractive summarization task.
def collator(items, cls_id, sep_id, pad_id, text_max_len):
    input_ids, labels, texts, summaries = zip(*items)

    input_ids = torch.stack([pad_1d(ids, text_max_len, pad_id) for ids in input_ids], dim=0)
    attn_mask = ~(input_ids == pad_id)

    cls_mask = input_ids == cls_id
    sep_mask = input_ids == sep_id

    # "Complete the Interval Segment Embeddings described in BertSum. 
    # Alternately assign a segment id of 0 or 1 to each sentence based on the odd or even nature of its position.
    tmp_input_ids = input_ids.view(-1, input_ids.size()[-1])
    segments_ids = []
    for content in tmp_input_ids:
        tmp = []
        _segs = [-1] + [i for i, t in enumerate(content) if t == sep_id]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                tmp += s * [0]
            else:
                tmp += s * [1]
        if len(tmp) < len(content):
            tmp += (len(content) - len(tmp)) * [0]
        segments_ids.append(tmp)
    segments_ids = torch.tensor(segments_ids).view(input_ids.size())

    labels = torch.tensor(sum(labels, list()))

    return Batch(
        input_ids=input_ids,
        attn_mask=attn_mask,
        cls_mask=cls_mask,
        sep_mask=sep_mask,
        seg=segments_ids,
        labels=labels,
        texts=texts,
        summaries=summaries,
    )


# The collator used by the dataloader for the three classification tasks: agnews, yahoo, dbpedia.
def gluecollator(items):
    input_ids, input_mask, segment_ids, label = zip(*items)

    input_ids = torch.stack(input_ids, dim=0)
    input_mask = torch.stack(input_mask, dim=0)
    segment_ids = torch.stack(segment_ids, dim=0)
    label = torch.stack(label, dim=0)

    return Batch(
        input_ids=input_ids,
        attn_mask=input_mask,
        seg=segment_ids,
        labels=label
    )
