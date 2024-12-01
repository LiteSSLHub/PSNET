import os

import torch.nn as nn

from pytorch_pretrained_bert.modeling import BertModel, BertConfig
from utils import CONFIG_NAME


# Get the BERT backbone model. If it is a teacher model, get the pre-trained bert-base-uncased; if it is a student model, initialize it randomly.
def get_model(config_name_or_path, mode_from_pretrained):
    if mode_from_pretrained:
        model = BertModel.from_pretrained(config_name_or_path)
    else:
        resolved_config_file = os.path.join(
            config_name_or_path, CONFIG_NAME)
        config = BertConfig.from_json_file(resolved_config_file)
        model = BertModel(config)
    return model


# Extractive summarization head
class ExtractiveSummaryHead(nn.Module):
    def __init__(self, hidden_size):
        super(ExtractiveSummaryHead, self).__init__()
        self.encoder = nn.Linear(hidden_size, 1)

    def forward(self, bert_output, cls_mask):
        # Select all hidden states corresponding to [CLS] in the sentence
        flatten_enc_sent_embs = bert_output.last_hidden_state.masked_select(cls_mask)
        enc_sent_embs = flatten_enc_sent_embs.view(-1, bert_output.last_hidden_state.size(-1))

        logits = self.encoder(enc_sent_embs).view(-1)
        return logits


# The classification task heads of the agnews, yahoo, and dbpedia dataset
class GlueTaskHead(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(GlueTaskHead, self).__init__()
        self.num_classes = num_classes
        self.encoder = nn.Linear(hidden_size, num_classes)

    def forward(self, bert_output):
        logits = self.encoder(bert_output.last_hidden_state[:, 0])
        return logits


# The main model class of the student model used by the PSNET framework
class ClusterSum(nn.Module):
    def __init__(self, config_name_or_path, student_num=2, fit_size=768, num_classes=1, extractive_summary=True,
                 model_from_pretrained=False):
        super(ClusterSum, self).__init__()
        self.student_num = student_num
        self.extractive_summary = extractive_summary
        self.student = nn.ModuleList([get_model(config_name_or_path, model_from_pretrained)
                                      for _ in range(self.student_num)])
        if self.extractive_summary:
            self.head = nn.ModuleList([ExtractiveSummaryHead(self.student[i].config.hidden_size)
                                       for i in range(self.student_num)])
        else:
            self.head = nn.ModuleList([GlueTaskHead(self.student[i].config.hidden_size, num_classes)
                                       for i in range(self.student_num)])
        # fit_dense is used to align different sized hidden_states of the teacher model and student model in the distillation step
        self.fit_dense = nn.ModuleList(
            [nn.Linear(self.student[i].config.hidden_size, fit_size) for i in range(self.student_num)])

    # Used during model inference. Only output model prediction.
    def predict(self, batch, specific_student):
        input_ids, attn_mask, seg = batch.input_ids, batch.attn_mask, batch.seg
        cls_mask = batch.cls_mask.unsqueeze(-1) if self.extractive_summary else None
        # Generate sentence feature
        output = self.student[specific_student](input_ids=input_ids,
                                                attention_mask=attn_mask,
                                                token_type_ids=seg)

        if self.extractive_summary:
            cluster_logit = self.head[specific_student](output, cls_mask)
        else:
            cluster_logit = self.head[specific_student](output)

        return cluster_logit

    def forward(self, batch, specific_student=None, inputs_embeds=None, distill_step=False, do_adversary=False):
        input_ids, attn_mask, seg = batch.input_ids, batch.attn_mask, batch.seg
        cls_mask = batch.cls_mask.unsqueeze(-1) if self.extractive_summary else None

        # Generate sentence representation
        all_encoder_layers = []
        all_encoder_atts = []
        all_logits = []
        all_embeds = []
        for i in range(self.student_num):
            if specific_student is None or i == specific_student:
                output = self.student[i](input_ids=input_ids if inputs_embeds is None else None,
                                         inputs_embeds=inputs_embeds,
                                         attention_mask=attn_mask,
                                         token_type_ids=seg,
                                         output_hidden_states=distill_step or do_adversary,
                                         output_attentions=distill_step)

                # Get all [cls] token representation
                if self.extractive_summary:
                    logits = self.head[i](output, cls_mask)
                else:
                    logits = self.head[i](output)
                all_logits.append(logits)
                if distill_step:
                    tmp = []
                    for sequence_layer in output.hidden_states:
                        tmp.append(self.fit_dense[i](sequence_layer))
                    all_encoder_layers.append(tmp)
                    all_encoder_atts.append(output.attentions)
                if do_adversary:
                    all_embeds.append(output.hidden_states[0])

        return all_logits, all_encoder_layers, all_encoder_atts, all_embeds


# The main model class of the teacher model used by the PSNET framework
class DynamicTeacher(nn.Module):
    def __init__(self, model_name_or_path, num_classes=1, extractive_summary=True):
        super(DynamicTeacher, self).__init__()
        self.extractive_summary = extractive_summary
        self.teacher = BertModel.from_pretrained(model_name_or_path)
        if self.extractive_summary:
            self.head = ExtractiveSummaryHead(self.teacher.config.hidden_size)
        else:
            self.head = GlueTaskHead(self.teacher.config.hidden_size, num_classes)

    def forward(self, batch, inputs_embeds=None, distill_step=False, do_adversary=False):
        input_ids, attn_mask, seg = batch.input_ids, batch.attn_mask, batch.seg
        cls_mask = batch.cls_mask.unsqueeze(-1) if self.extractive_summary else None

        # Generate sentence representation
        output = self.teacher(input_ids=input_ids if inputs_embeds is None else None,
                              inputs_embeds=inputs_embeds,
                              attention_mask=attn_mask,
                              token_type_ids=seg,
                              output_hidden_states=distill_step or do_adversary,
                              output_attentions=distill_step)

        # Get all [cls] token representation
        if self.extractive_summary:
            logits = self.head(output, cls_mask)
        else:
            logits = self.head(output)
        encoder_layers = output.hidden_states
        encoder_atts = output.attentions
        embeds = output.hidden_states[0] if output.hidden_states is not None else None

        return logits, encoder_layers, encoder_atts, embeds
