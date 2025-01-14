import re

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from utils import get_logger
from metrics import (
    calc_rouge_from_pyrouge,
    calc_rouge_from_python_implementation,
    simple_accuracy
)

log = get_logger(__name__)


# Extract the original sentences from the logits output by the model as the result of the extractive summary
def extract_oracle_from_logits(logits, texts, summaries, extract_n_sents, trigram_block=True):
    def text_clean(s):
        return re.sub(r'[^a-zA-Z0-9 ]', '', s)

    # Extract all n-gram phrases in the sentence.
    def get_ngrams(n, text):
        if isinstance(text, str):
            text = text_clean(text)
            text = text.split()

        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i: i + n]))
        return ngram_set

    # Check whether there is a repetition of 3-gram phrases between the current candidate sentence and 
    # the sentences that have already been extracted. If there is a repetition, discard the candidate sentence
    def has_same_trigram(c, p):
        tri_c = get_ngrams(3, text_clean(c).split())

        if isinstance(p, str):
            p = [p]

        for i in range(len(p)):
            tri_p = get_ngrams(3, text_clean(p[i]).split())
            if len(tri_c.intersection(tri_p)) > 0:
                return True

        return False

    hypothesis, references = list(), list()

    for logits, text, summary in zip(logits, texts, summaries):
        scores = torch.sigmoid(logits)
        _, sort_idxs = torch.sort(scores, descending=True)

        cur_hypo = list()
        for idx in sort_idxs:
            sent_to_check = text[idx].strip()

            if trigram_block:
                if not has_same_trigram(sent_to_check, cur_hypo):
                    cur_hypo.append(sent_to_check)
            else:
                cur_hypo.append(sent_to_check)

            if len(cur_hypo) == extract_n_sents:
                break

        hypothesis.append("\n".join(cur_hypo))
        references.append("\n".join(summary))

    return hypothesis, references


# The evaluation function for extractive summarization
def sumeval(model, dataloader, device, specific_student, extract_nsents, is_student=True, pyrouge=True,
            trigram_block=True):
    model.eval()

    eval_loss = 0.0
    hypothesis, references = list(), list()
    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            if is_student:
                logits = model.predict(batch, specific_student=specific_student)
            else:
                logits, _, _, _ = model(batch)

        loss = F.binary_cross_entropy_with_logits(logits, batch.labels)
        eval_loss += loss.data

        nsents_per_item = torch.sum(batch.cls_mask, dim=1).data.tolist()
        splited_logits = torch.split(logits, nsents_per_item, dim=0)
        hypos, refer = extract_oracle_from_logits(splited_logits,
                                                  batch.texts,
                                                  batch.summaries,
                                                  extract_nsents,
                                                  trigram_block=trigram_block)
        hypothesis.extend(hypos)
        references.extend(refer)

    eval_loss /= len(dataloader)
    if pyrouge:
        rouge_scores = calc_rouge_from_pyrouge(hypothesis, references)
    else:
        rouge_scores = calc_rouge_from_python_implementation(hypothesis, references)

    output_note = "********** Evaluate end "
    output_note += "| " + "eval loss" + f" {eval_loss:5.4f} "
    for key in rouge_scores:
        output_note += "| " + key + f" {rouge_scores[key]:5.2f} "
    output_note += "**********"
    log.info(output_note)

    return eval_loss, rouge_scores


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name in ["yahoo", "dbpedia", "agnews", "ag_news", "amazon_review", "yahoo_answers", "yelp_review"]:
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


# The evaluation function for the three classification tasks: agnews, yahoo, dbpedia.
def glueeval(model, dataloader, device, specific_student, task_name=None, is_student=True, num_labels=2):
    model.eval()
    eval_loss = 0.0
    preds = []
    labels = []

    for batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            if is_student:
                logits = model.predict(batch, specific_student=specific_student)
            else:
                logits, _, _, _ = model(batch)

        # create eval loss and other metric required by the task
        loss_fct = nn.CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), batch.labels.view(-1))

        labels.append(batch.labels)
        eval_loss += tmp_eval_loss.item()
        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / len(dataloader)

    labels = torch.cat(labels, dim=0)
    preds = preds[0]
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(task_name, preds, labels.cpu().numpy())

    output_note = "********** Evaluate end "
    output_note += "| " + "eval loss" + f" {eval_loss:5.4f} "
    for key in result:
        output_note += "| " + key + f" {result[key]:5.4f} "
    output_note += "**********"
    log.info(output_note)

    return eval_loss, result


def evaluate(model, dataloader, specific_student, extract_nsents, device, task_name=None, is_student=True, pyrouge=True,
             trigram_block=True):
    if task_name == "sum":
        return sumeval(model,
                       dataloader,
                       device,
                       specific_student=specific_student,
                       extract_nsents=extract_nsents,
                       is_student=is_student,
                       pyrouge=pyrouge,
                       trigram_block=trigram_block)
    else:
        return glueeval(model,
                        dataloader,
                        device,
                        specific_student=specific_student,
                        task_name=task_name,
                        is_student=is_student,
                        num_labels=dataloader.dataset.n_labels + 1)
