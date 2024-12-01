import os
import traceback
from argparse import ArgumentParser
import random
import math

import numpy as np
import torch
import torch.nn as nn
from torch import linalg as LA
from transformers.utils.logging import set_verbosity_error
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F

from evaluate import evaluate
from model import ClusterSum, DynamicTeacher
from data import ExtractiveSummaryDataModule, SelfUdaDataModule
from utils import (
    load_checkpoints,
    save_checkpoints,
    get_logger
)

log = get_logger(__name__)


def parse_args():
    parser = ArgumentParser()
    # training arguments
    subparser = parser.add_argument_group("train")
    subparser.add_argument("--gpu", type=int, default=0, help="GPU ID to use.")
    subparser.add_argument("--cuda", action="store_true", help="GPU or CPU.")
    subparser.add_argument("--seed", type=int, default=0, help="Random seed.")
    subparser.add_argument("--root_dir", type=str, default="./experiments/cnndm",
                           help="The root directory of this run.")
    subparser.add_argument("--ckpt_dir", type=str, default="checkpoints",
                           help="The specific directory name in the root directory to save checkpoints.")
    subparser.add_argument("--test_after_train", action="store_true", default=True, help="Do test after training.")
    subparser.add_argument("--do_cluster", action="store_true", help="Cluster training or not.")
    subparser.add_argument("--do_normal", action="store_true", help="Normal training or not.")
    subparser.add_argument("--do_distill", action="store_true", help="Distill training or not.")
    subparser.add_argument("--task_name", type=str, default="sum", help="specific task name.")
    subparser.add_argument("--supervised_size", type=int, default=50000,
                           help="Number of supervised data in consistency training.")
    subparser.add_argument("--unsupervised_size", type=int, default=50000,
                           help="Number of unsupervised data in consistency training.")
    subparser.add_argument("--student_promotion_lr", type=float, default=2e-3, help="Student_promotion learning rate.")
    subparser.add_argument("--student_distill_lr", type=float, default=5e-5, help="Student_distill learning rate.")
    subparser.add_argument("--student_mutual_lr", type=float, default=1e-4, help="Student_mutual learning rate.")
    subparser.add_argument("--teacher_lr", type=float, default=2e-3, help="Teacher learning rate.")
    subparser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay of adamW.")
    subparser.add_argument("--num_training_steps", type=int, default=50000, help="Total number of training steps.")
    subparser.add_argument("--promotion_warmup_proportion", default=0.2, type=float,
                           help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    subparser.add_argument("--distill_warmup_proportion", default=0.1, type=float,
                           help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    subparser.add_argument("--val_interval", type=int, default=2000, help="Intervals for evaluate.")
    subparser.add_argument("--resume_ckpt_path", type=str, default=None, help="Resume checkpoint path.")
    subparser.add_argument("--do_adversary_1", action="store_true", help="generate adversary data or not.")
    subparser.add_argument("--do_adversary_2", action="store_true", help="generate adversary data or not.")
    subparser.add_argument("--do_adversary_3", action="store_true", help="generate adversary data or not.")
    subparser.add_argument("--do_adversary_teacher", action="store_true", help="generate adversary data or not.")
    subparser.add_argument("--do_block", action="store_true", default=True, help="Trigram block or not.")
    subparser.add_argument("--lambdau", type=float, default=10,
                           help="Hyperparameters representing the importance of mutual learning.")
    subparser.add_argument("--belittle", type=float, default=0.8,
                           help="Proportion of teachers with supervision labels in training.")
    subparser.add_argument("--rampup_rate", type=float, default=0.2,
                           help="Proportion of training to perform ramp up step.")
    subparser.add_argument("--distill_mode", type=int, default=0,
                           help="Distill mode.")
    subparser.add_argument("--static_adv_iter", action="store_true", help="Whether use static adversarial step.")
    subparser.add_argument("--adv_iter", type=int, default=1, help="Static adversarial step period.")


    # dataset arguments
    subparser = parser.add_argument_group("data")
    subparser.add_argument("--cnndm_dataset_name", type=str, required=True, help="Name of cnn_dailymail dataset.")
    subparser.add_argument("--glue_dataset_name", type=str, help="Name of glue dataset.")
    subparser.add_argument("--train_batch_size", type=int, default=32, help="Batch size of training.")
    subparser.add_argument("--val_batch_size", type=int, default=32, help="Batch size of validation.")
    subparser.add_argument("--test_batch_size", type=int, default=32, help="Batch size of testing.")
    subparser.add_argument("--tokenizer_name_or_path", type=str, default="bert-base-uncased",
                           # choices=["bert-base-uncased", "bert-large-uncased"],
                           help="The name or path of pretrained tokenizer.")
    subparser.add_argument("--num_workers", type=int, default=8, help="Number of process workers in dataloader.")
    subparser.add_argument("--extract_nsents", type=int, default=3, help="Number of oracle summary.")

    # model arguments
    subparser = parser.add_argument_group("model")
    subparser.add_argument("--config_name_or_path", type=str, required=True,
                           help="Location of config file, or model name if use from_pretrained to get config file.")
    subparser.add_argument("--teacher_model", type=str, required=True, help="Path or model name to get teacher model.")
    subparser.add_argument("--student_num", type=int, default=2, help="Number of student model.")
    subparser.add_argument("--model_from_pretrained", action="store_true",
                           help="Use from_pretrained to get model or not.")
    args = parser.parse_args()
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# The linear ramp-up function for the consistency loss part starts at 0, increases linearly, 
# and remains at 1 once the number of training steps reaches ramp_up_rate * total_step.
def linear_rampup(current, args):
    rampup_length = args.num_training_steps * args.rampup_rate
    current = np.clip(current / rampup_length, 0.0, 1.0)
    return float(current)


def get_adv_iter(current, args):
    if args.static_adv_iter:
        return args.adv_iter
    else:
        return (current // 10000)


# Data augmentation method, where the result of the augmentation is in the form of input_embeds rather than the original form of token_ids. 
# It's only necessary to know that this part is about data augmentation, the content details need not be looked at.
def vat_generator(model, x, specific_student, inputs_embeds, target_logits, iter, device=None, loss_func=None,
                  is_teacher=False, adv_step_size=1e-3, adv_epsilon=1e-6, adv_noise_var=1e-5):
    if iter == 0:
        return inputs_embeds
    inputs_embeds = inputs_embeds.detach()
    target_logits = target_logits.detach()
    noise = (inputs_embeds.data.new(inputs_embeds.size()).normal_(0, 1) * adv_noise_var).to(device)
    noise.detach()
    noise.requires_grad_()
    loss = nn.MSELoss(reduction='sum') if loss_func is None else loss_func
    for step in range(iter):
        if is_teacher:
            adv_logits, _, _, _ = model(batch=x, inputs_embeds=inputs_embeds + noise)
        else:
            adv_logits, _, _, _ = model(batch=x, specific_student=specific_student, inputs_embeds=inputs_embeds + noise)
            adv_logits = adv_logits[0]
        adv_loss = loss(adv_logits, target_logits)
        delta_grad, = torch.autograd.grad(adv_loss, noise, retain_graph=False)
        norm = delta_grad.norm()
        if torch.isnan(norm) or torch.isinf(norm):
            return inputs_embeds
        delta_grad = noise + delta_grad * adv_step_size
        noise = delta_grad / (LA.norm(delta_grad, dim=-1, keepdim=True) + adv_epsilon)  # , ord=float('inf')
        noise = noise.detach()
        noise.requires_grad_()
    return inputs_embeds + noise


# Invoke the data augmentation function, providing it with different loss functions and labels 
# according to different tasks and data types for data augmentation. The content details need not be looked at.
def adversary(batch, model, student_num, device, label=None, iter=1, task_name=None, is_teacher=False):
    if task_name == "sum":
        loss_func = nn.BCEWithLogitsLoss(reduction='sum')
    else:
        loss_func = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        if is_teacher:
            logits, _, _, initial_embeds = model(batch, do_adversary=True)
        else:
            logits, _, _, student_embeds = model(batch, do_adversary=True, specific_student=student_num)
            logits = logits[0]
            initial_embeds = student_embeds[0]
    adversary_embeds = vat_generator(model, batch, student_num, initial_embeds,
                                     logits if label is None else label, iter,
                                     device, None if label is None else loss_func, is_teacher)
    return adversary_embeds


# Obtain the optimizer and scheduler needed during the training process.
def get_optimizer_and_scheduler(model, args, is_teacher=False):
    def lr_lambda(current_step):
        current_step = current_step + 1
        return min(current_step ** -0.5,
                   current_step * ((args.num_training_steps * args.promotion_warmup_proportion) ** -1.5))

    def get_decay_parameter(submodel):
        # Prepare optimizer
        param_optimizer = list(submodel.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return optimizer_grouped_parameters
    
    if is_teacher:
        optimizer = torch.optim.AdamW(get_decay_parameter(model), lr=args.teacher_lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler
    else:
        promotion_optimizer = []
        distill_optimizer = []
        mutual_optimizer = []
        promotion_scheduler = []
        distill_scheduler = []
        mutual_scheduler = []
        for n, p in model.named_children():
            if n == "student":
                student = p
            elif n == "head":
                head = p
            elif n == "fit_dense":
                fit_dense = p

        student_list = list(student.children())
        head_list = list(head.children())
        fit_dense_list = list(fit_dense.children())
        for i in range(args.student_num):
            promotion_optimizer.append(torch.optim.AdamW(
                get_decay_parameter(student_list[i]) + get_decay_parameter(head_list[i]),
                lr=args.student_promotion_lr))
            distill_optimizer.append(
                torch.optim.AdamW(get_decay_parameter(student_list[i]) + get_decay_parameter(fit_dense_list[i]) +
                                  get_decay_parameter(head_list[i]),
                                  lr=args.student_distill_lr))
            mutual_optimizer.append(torch.optim.AdamW(
                get_decay_parameter(student_list[i]) + get_decay_parameter(head_list[i]),
                lr=args.student_mutual_lr))
            promotion_scheduler.append(torch.optim.lr_scheduler.LambdaLR(promotion_optimizer[i], lr_lambda))
            distill_scheduler.append(
                get_linear_schedule_with_warmup(distill_optimizer[i],
                                                num_warmup_steps=args.num_training_steps * args.distill_warmup_proportion,
                                                num_training_steps=args.num_training_steps))
            mutual_scheduler.append(
                get_linear_schedule_with_warmup(mutual_optimizer[i],
                                                num_warmup_steps=args.num_training_steps * args.distill_warmup_proportion,
                                                num_training_steps=args.num_training_steps))
        return promotion_optimizer, promotion_scheduler, distill_optimizer, distill_scheduler, mutual_optimizer, mutual_scheduler


def cluster_train(args):
    log.info(f'args:{args}')

    # Obtain different data modules according to the different tasks.
    if args.task_name == "sum":
        datamodule = ExtractiveSummaryDataModule(args)
        datamodule.prepare(args)
        num_labels = 1
    elif args.task_name in ["yahoo", "dbpedia", "agnews", "ag_news", "amazon_review", "yahoo_answers", "yelp_review"]:
        datamodule = SelfUdaDataModule(args)
        num_labels = datamodule.num_labels

    # Initialize the teacher model
    teacher_model = DynamicTeacher(model_name_or_path=args.teacher_model,
                                   num_classes=num_labels,
                                   extractive_summary=args.task_name == "sum")
    # Initialize the student model.
    student_model = ClusterSum(config_name_or_path=args.config_name_or_path,
                               student_num=args.student_num,
                               fit_size=teacher_model.teacher.config.hidden_size,
                               num_classes=num_labels,
                               extractive_summary=args.task_name == "sum",
                               model_from_pretrained=args.model_from_pretrained)
    
    # Obtain the optimizer and scheduler needed for the student model. More optimizers and schedulers are needed 
    # here because we divide the supervised learning part, the distillation part, and the mutual learning part into 
    # three sections for optimization, which makes it convenient to set appropriate learning rates separately.
    student_promotion_optimizer, student_promotion_scheduler, student_distill_optimizer, student_distill_scheduler, student_mutual_optimizer, student_mutual_scheduler = get_optimizer_and_scheduler(
        student_model, args, is_teacher=False)
    # Obtain the optimizer and scheduler for the teacher model
    teacher_optimizer, teacher_scheduler = get_optimizer_and_scheduler(teacher_model, args, is_teacher=True)

    # Obtain the dataloader needed for training and evaluation.
    train_dataloader, unsupervised_dataloader = datamodule.train_dataloader(), datamodule.unsupervised_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    train_dataiter = iter(train_dataloader)
    unsupervised_dataiter = iter(unsupervised_dataloader)

    # Obtain device information. Use GPU for training if available, otherwise use CPU.
    device = torch.device("cpu")
    if torch.cuda.is_available():
        log.info(f"Use gpu: {args.gpu}")
        device = torch.device(args.gpu)
        student_model = student_model.to(torch.device(args.gpu))
        teacher_model = teacher_model.to(torch.device(args.gpu))

    # Select different cross-entropy loss functions according to classification tasks and summarization tasks.
    # BCE_Loss is a loss function specialized for binary classification, used in our extractive summarization task.
    loss_MSE = nn.MSELoss()
    # loss_KL = nn.KLDivLoss(reduction="none")
    if args.task_name == "sum":
        loss_CE = nn.BCEWithLogitsLoss()
    else:
        loss_CE = nn.CrossEntropyLoss()

    # The definition of the process information that needs to be recorded during training.
    current_step = 0
    promotion_loss_dic = [{'sup_loss': 0.0} for _ in range(args.student_num)]
    distill_loss_dic = [{'total_loss': 0.0, 'att_loss': 0.0, 'rep_loss': 0.0, 'mutual_loss': 0.0, 'logits_loss': 0.0}
                        for _ in range(args.student_num)]
    teacher_loss_dic = {'total_loss': 0.0, 'sup_loss': 0.0, 'belittle_loss': 0.0}
    best_eval_loss = [np.inf] * args.student_num
    best_metric_score = [np.NINF] * args.student_num
    best_loss_checkpoints_filename = [""] * args.student_num
    best_metric_checkpoints_filename = [""] * args.student_num

    log.info("***** Running training *****")
    log.info("  Supervised Batch size = %d", train_dataloader.batch_size)
    log.info("  Unsupervised Batch size = %d", unsupervised_dataloader.batch_size)
    log.info("  Num steps = %d", args.num_training_steps)

    while current_step < args.num_training_steps:
        teacher_model.train()
        student_model.train()

        # log.info("1:{}".format(torch.cuda.memory_allocated(0)))
        # Obtain the batch of annotated data.
        try:
            batch = next(train_dataiter)
        except StopIteration:
            train_dataiter = iter(train_dataloader)
            batch = next(train_dataiter)
        except Exception as e:
            log.error(f"Error when loading data: {e}")
            log.error(traceback.format_exc())
            exit()

        # Obtain the batch of unlabeled data.
        try:
            unsupervised_batch = next(unsupervised_dataiter)
        except StopIteration:
            unsupervised_dataiter = iter(unsupervised_dataloader)
            unsupervised_batch = next(unsupervised_dataiter)
        except Exception as e:
            log.error(f"Error when loading data: {e}")
            log.error(traceback.format_exc())
            exit()

        # Transfer the data in the batch to the corresponding device obtained earlier
        batch, unsupervised_batch = batch.to(device), unsupervised_batch.to(device)

        # log.info("2:{}".format(torch.cuda.memory_allocated(0)))

        # ***** Teacher and student promotion step *****
        # The student model optimizes using the labeled data.
        all_student_logits = []
        for bias in range(args.student_num):
            # Invoke the data augmentation method.
            input_embeds = None
            if args.do_adversary_1:
                input_embeds = adversary(batch, student_model, bias, device, batch.labels,
                                         iter=get_adv_iter(current_step, args), task_name=args.task_name)

            student_logits, _, _, _ = student_model(batch,
                                                    inputs_embeds=input_embeds,
                                                    specific_student=bias,
                                                    distill_step=False)
            all_student_logits.append(student_logits[0].detach())
            sup_loss = loss_CE(student_logits[0], batch.labels)
            promotion_loss_dic[bias]['sup_loss'] += sup_loss.item()

            # Update the gradients of the student model.
            student_promotion_optimizer[bias].zero_grad()
            sup_loss.backward()
            student_promotion_optimizer[bias].step()
            student_promotion_scheduler[bias].step()

        # The teacher model optimizes using the labeled data.
        # Invoke the data augmentation method.
        input_embeds = None
        if args.do_adversary_teacher:
            input_embeds = adversary(batch, teacher_model, -1, device, batch.labels,
                                     iter=get_adv_iter(current_step, args), task_name=args.task_name, is_teacher=True)

        teacher_logits, _, _, _ = teacher_model(batch, inputs_embeds=input_embeds, distill_step=False)
        belittle_loss = 0.0
        for bias in range(args.student_num):
            if args.task_name == "sum":
                tmp_student_logits = torch.sigmoid(all_student_logits[bias].detach())
            else:
                tmp_student_logits = torch.softmax(all_student_logits[bias].detach(), dim=-1)
            belittle_loss += (1 - args.belittle) * loss_CE(teacher_logits, tmp_student_logits)
        sup_loss = args.belittle * loss_CE(teacher_logits, batch.labels)
        promotion_loss = sup_loss + belittle_loss / args.student_num

        teacher_loss_dic['total_loss'] += promotion_loss.item()
        teacher_loss_dic['sup_loss'] += sup_loss.item()
        teacher_loss_dic['belittle_loss'] += belittle_loss.item() / args.student_num

        # Update the gradients of the teacher model.
        teacher_optimizer.zero_grad()
        promotion_loss.backward()
        teacher_optimizer.step()
        teacher_scheduler.step()

        # log.info("3:{}".format(torch.cuda.memory_allocated(0)))

        # ***** Student distillation step *****
        # Using unsupervised data, perform feature-level distillation from the teacher model to the student model. 
        # The following code is similar to TinyBert.
        with torch.no_grad():
            teacher_logits, teacher_encoder_layers, teacher_encoder_atts, _ = teacher_model(unsupervised_batch,
                                                                                            distill_step=True)

        teacher_layer_num = len(teacher_encoder_atts)

        for bias in range(args.student_num):
            input_embeds = None
            # Invoke the data augmentation method.
            if args.do_adversary_2:
                input_embeds = adversary(unsupervised_batch, student_model, bias, device,
                                         iter=get_adv_iter(current_step, args), task_name=args.task_name)
            _, student_encoder_layers, student_encoder_atts, _ = student_model(unsupervised_batch,
                                                                               inputs_embeds=input_embeds,
                                                                               specific_student=bias,
                                                                               distill_step=True)
            student_layer_num = len(student_encoder_atts[0])
            assert teacher_layer_num % student_layer_num == 0
            layers_per_block = int(teacher_layer_num / student_layer_num)

            # [3,6,9,12],[2,5,8,12],[1,4,7,12]
            if args.distill_mode == 0:
                att_loss = 0.0
                new_teacher_atts = [teacher_encoder_atts[i * layers_per_block + layers_per_block - 1 - bias] for i in
                                    range(student_layer_num - 1)] + [teacher_encoder_atts[teacher_layer_num - 1]]
                for student_att, teacher_att in zip(student_encoder_atts[0], new_teacher_atts):
                    att_loss += loss_MSE(student_att, teacher_att)

                rep_loss = 0.0
                new_teacher_reps = [teacher_encoder_layers[0]] + \
                                   [teacher_encoder_layers[(i + 1) * layers_per_block - bias] for i in
                                    range(student_layer_num - 1)] + \
                                   [teacher_encoder_layers[teacher_layer_num]]
                for student_rep, teacher_rep in zip(student_encoder_layers[0], new_teacher_reps):
                    rep_loss += loss_MSE(student_rep, teacher_rep)

            # [1，2，3，4],[5，6，7，8],[9，10，11，12]
            elif args.distill_mode == 1:
                tmpbias = bias if bias < args.student_num // 2 else layers_per_block - args.student_num + bias
                att_loss = 0.0
                new_teacher_atts = [teacher_encoder_atts[tmpbias * student_layer_num + i] for
                                    i in range(student_layer_num)]
                for student_att, teacher_att in zip(student_encoder_atts[0], new_teacher_atts):
                    att_loss += loss_MSE(student_att, teacher_att)

                rep_loss = 0.0
                new_teacher_reps = [teacher_encoder_layers[tmpbias * student_layer_num + i] for i in
                                    range(student_layer_num + 1)]
                for student_rep, teacher_rep in zip(student_encoder_layers[0], new_teacher_reps):
                    rep_loss += loss_MSE(student_rep, teacher_rep)

            # [3,6,9,12],[2,5,8,11],[1,4,7,10]
            elif args.distill_mode == 2:
                att_loss = 0.0
                new_teacher_atts = [teacher_encoder_atts[i * layers_per_block + layers_per_block - 1 - bias]
                                    for i in range(student_layer_num)]
                for student_att, teacher_att in zip(student_encoder_atts[0], new_teacher_atts):
                    att_loss += loss_MSE(student_att, teacher_att)

                rep_loss = 0.0
                new_teacher_reps = [teacher_encoder_layers[0]] + \
                                   [teacher_encoder_layers[(i + 1) * layers_per_block - bias] for i in
                                    range(student_layer_num)]
                for student_rep, teacher_rep in zip(student_encoder_layers[0], new_teacher_reps):
                    rep_loss += loss_MSE(student_rep, teacher_rep)

            # distill_loss = linear_rampup(current_step, args) * (att_loss + rep_loss)
            distill_loss = (att_loss + rep_loss)
            distill_loss_dic[bias]['total_loss'] += distill_loss.item()
            distill_loss_dic[bias]['att_loss'] += att_loss.item()
            distill_loss_dic[bias]['rep_loss'] += rep_loss.item()

            # Update the gradients of the student model.
            student_distill_optimizer[bias].zero_grad()
            distill_loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.)
            student_distill_optimizer[bias].step()
            student_distill_scheduler[bias].step()

        # log.info("4:{}".format(torch.cuda.memory_allocated(0)))
        # Using unsupervised data, complete the distillation of predictions between the teacher model 
        # and the student model, as well as the mutual learning between students.
        all_unsupervised_logits = []
        for bias in range(args.student_num):
            input_embeds = None
            # Invoke the data augmentation method.
            if args.do_adversary_3:
                input_embeds = adversary(unsupervised_batch, student_model, bias, device,
                                         iter=get_adv_iter(current_step, args), task_name=args.task_name)
            student_logits, _, _, _ = student_model(unsupervised_batch,
                                                    inputs_embeds=input_embeds,
                                                    distill_step=False,
                                                    specific_student=bias)
            all_unsupervised_logits.append(student_logits[0])

        for bias in range(args.student_num):
            # Learn from the predictions of the teacher model
            logits_loss = loss_MSE(all_unsupervised_logits[bias], teacher_logits)
            # Learn from the predictions of other student models.
            mutual_loss = 0.0
            for other_student in range(args.student_num):
                if other_student != bias:
                    mutual_loss += loss_MSE(all_unsupervised_logits[bias],
                                            all_unsupervised_logits[other_student].detach())

            prediction_mutual_loss = logits_loss
            prediction_mutual_loss += args.lambdau * linear_rampup(current_step, args) * \
                                      mutual_loss / (args.student_num - 1) if args.student_num > 1 else 0

            distill_loss_dic[bias]['total_loss'] += prediction_mutual_loss.item()
            distill_loss_dic[bias]['mutual_loss'] += args.lambdau * linear_rampup(current_step, args) * \
                                                     mutual_loss.item() / \
                                                     (args.student_num - 1) if args.student_num > 1 else 0
            distill_loss_dic[bias]['logits_loss'] += logits_loss.item()

            # Update the gradients of the student model
            student_mutual_optimizer[bias].zero_grad()
            prediction_mutual_loss.backward()
            student_mutual_optimizer[bias].step()
            student_mutual_scheduler[bias].step()

        current_step = current_step + 1

        # Output the process information that needs to be recorded during training
        if current_step % 100 == 0:
            log.info(f"Step {current_step:3d}")
            for i in range(args.student_num):
                log.info(
                    f"Student {i} in promotion stage | sup loss {(promotion_loss_dic[i]['sup_loss'] / 100.0):5.4f}")
                log.info(
                    f"Student {i} in distill stage | total loss {(distill_loss_dic[i]['total_loss'] / 100.0):5.4f} | att loss {(distill_loss_dic[i]['att_loss'] / 100.0):5.4f} | rep loss {(distill_loss_dic[i]['rep_loss'] / 100.0):5.4f} | logits loss {(distill_loss_dic[i]['logits_loss'] / 100.0):5.4f} | mutual loss {(distill_loss_dic[i]['mutual_loss'] / 100.0):5.4f}")
                promotion_loss_dic[i]['sup_loss'] = 0.0
                distill_loss_dic[i]['total_loss'] = 0.0
                distill_loss_dic[i]['att_loss'] = 0.0
                distill_loss_dic[i]['rep_loss'] = 0.0
                distill_loss_dic[i]['logits_loss'] = 0.0
                distill_loss_dic[i]['mutual_loss'] = 0.0
            log.info(
                f"Teacher | total loss {(teacher_loss_dic['total_loss'] / 100.0):5.4f} | sup loss {(teacher_loss_dic['sup_loss'] / 100.0):5.4f} | belittle loss {(teacher_loss_dic['belittle_loss'] / 100.0):5.4f}")
            teacher_loss_dic['total_loss'] = 0.0
            teacher_loss_dic['sup_loss'] = 0.0
            teacher_loss_dic['belittle_loss'] = 0.0

        # Evaluate the performance on the validation set.
        if current_step % args.val_interval == 0:
            for bias in range(args.student_num):
                eval_loss, metric_result = evaluate(student_model,
                                                    val_dataloader,
                                                    bias,
                                                    args.extract_nsents,
                                                    device,
                                                    task_name=args.task_name,
                                                    is_student=True,
                                                    pyrouge=False,
                                                    trigram_block=args.do_block)

                checkpoints = {
                    "step": current_step,
                    "model": student_model.state_dict(),
                }
                checkpoints_filename = os.path.join(args.root_dir, args.ckpt_dir, f"model_step_{current_step}.ckpt")
                save_checkpoints(checkpoints_filename, checkpoints)

                if eval_loss < best_eval_loss[bias]:
                    best_eval_loss[bias] = eval_loss
                    best_loss_checkpoints_filename[bias] = checkpoints_filename
                tmp_metric_score = sum(metric_result.values())
                if tmp_metric_score > best_metric_score[bias]:
                    best_metric_score[bias] = tmp_metric_score
                    best_metric_checkpoints_filename[bias] = checkpoints_filename

    log.info("Train end.")

    if args.test_after_train:
        for bias in range(args.student_num):
            log.info(f"For student {bias}, the best loss checkpoint file is in {best_loss_checkpoints_filename[bias]}")
            log.info(
                f"For student {bias}, the best rouge checkpoint file is in {best_metric_checkpoints_filename[bias]}")
            ckpt = load_checkpoints(best_loss_checkpoints_filename[bias], device)
            student_model.load_state_dict(ckpt["model"])
            log.info("Test the best loss checkpoints.")
            evaluate(student_model,
                     test_dataloader,
                     bias,
                     args.extract_nsents,
                     device,
                     task_name=args.task_name,
                     is_student=True,
                     pyrouge=True,
                     trigram_block=args.do_block)
            ckpt = load_checkpoints(best_metric_checkpoints_filename[bias], device)
            student_model.load_state_dict(ckpt["model"])
            log.info("Test the best metric checkpoints.")
            evaluate(student_model,
                     test_dataloader,
                     bias,
                     args.extract_nsents,
                     device,
                     task_name=args.task_name,
                     is_student=True,
                     pyrouge=True,
                     trigram_block=args.do_block)

    return best_loss_checkpoints_filename, best_metric_checkpoints_filename


# Simply fine-tune using a single model, which is the most basic training, as one of the baselines.
def normal_train(args):
    log.info(f'args:{args}')

    if args.task_name == "sum":
        datamodule = ExtractiveSummaryDataModule(args)
        datamodule.prepare(args)
        num_labels = 1
    elif args.task_name in ["yahoo", "dbpedia", "agnews", "ag_news", "amazon_review", "yahoo_answers", "yelp_review"]:
        datamodule = SelfUdaDataModule(args)
        num_labels = datamodule.num_labels

    model = DynamicTeacher(model_name_or_path=args.teacher_model,
                           num_classes=num_labels,
                           extractive_summary=args.task_name == "sum")

    optimizer, scheduler = get_optimizer_and_scheduler(model, args, is_teacher=True)

    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    train_dataiter = iter(train_dataloader)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        log.info(f"Use gpu: {args.gpu}")
        device = torch.device(args.gpu)
        model = model.to(torch.device(args.gpu))

    if args.task_name == "sum":
        loss_CE = nn.BCEWithLogitsLoss()
    else:
        loss_CE = nn.CrossEntropyLoss()
    current_step = 0

    loss_dic = {'sup_loss': 0.0}
    best_eval_loss = np.inf
    best_metric_score = np.NINF
    best_loss_checkpoints_filename = None
    best_metric_checkpoints_filename = None
    log.info("***** Running training *****")
    log.info("  Supervised Batch size = %d", train_dataloader.batch_size)
    log.info("  Num steps = %d", args.num_training_steps)

    while current_step < args.num_training_steps:
        model.train()

        # log.info("1:{}".format(torch.cuda.memory_allocated(0)))

        try:
            batch = next(train_dataiter)
        except StopIteration:
            train_dataiter = iter(train_dataloader)
            batch = next(train_dataiter)
        except Exception as e:
            log.error(f"Error when loading data: {e}")
            log.error(traceback.format_exc())
            exit()

        batch = batch.to(device)

        # log.info("2:{}".format(torch.cuda.memory_allocated(0)))

        # ***** Normal supervised training *****
        logits, _, _, _ = model(batch, distill_step=False)

        sup_loss = loss_CE(logits, batch.labels)

        loss_dic['sup_loss'] += sup_loss.item()
        optimizer.zero_grad()
        sup_loss.backward()
        optimizer.step()
        scheduler.step()

        current_step = current_step + 1

        if current_step % 100 == 0:
            log.info(f"Step {current_step:3d}")
            log.info(
                f"Model in normal train | sup loss {(loss_dic['sup_loss'] / 100.0):5.4f}")
            loss_dic['sup_loss'] = 0.0

        if current_step % args.val_interval == 0:
            eval_loss, metric_result = evaluate(model,
                                                val_dataloader,
                                                0,
                                                args.extract_nsents,
                                                device,
                                                task_name=args.task_name,
                                                is_student=False,
                                                pyrouge=False,
                                                trigram_block=args.do_block)

            checkpoints = {
                "step": current_step,
                "model": model.state_dict(),
            }
            checkpoints_filename = os.path.join(args.root_dir, args.ckpt_dir, f"model_step_{current_step}.ckpt")
            save_checkpoints(checkpoints_filename, checkpoints)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_loss_checkpoints_filename = checkpoints_filename
            tmp_metric_score = sum(metric_result.values())
            if tmp_metric_score > best_metric_score:
                best_metric_score = tmp_metric_score
                best_metric_checkpoints_filename = checkpoints_filename

    log.info("Train end.")
    log.info(f"The best loss checkpoint file is in {best_loss_checkpoints_filename}")
    log.info(f"The best rouge checkpoint file is in {best_metric_checkpoints_filename}")

    if args.test_after_train:
        ckpt = load_checkpoints(best_loss_checkpoints_filename, device)
        model.load_state_dict(ckpt["model"])
        log.info("Test the best loss checkpoints.")
        evaluate(model,
                 test_dataloader,
                 0,
                 args.extract_nsents,
                 device,
                 task_name=args.task_name,
                 is_student=False,
                 pyrouge=True,
                 trigram_block=args.do_block)
        ckpt = load_checkpoints(best_metric_checkpoints_filename, device)
        model.load_state_dict(ckpt["model"])
        log.info("Test the best rouge checkpoints.")
        evaluate(model,
                 test_dataloader,
                 0,
                 args.extract_nsents,
                 device,
                 task_name=args.task_name,
                 is_student=False,
                 pyrouge=True,
                 trigram_block=args.do_block)

    return best_loss_checkpoints_filename, best_metric_checkpoints_filename


# Directly use the fine-tuned teacher model for task-specific distillation, 
# without using online distillation or mutual learning, as one of the baselines.
def distill_train(args):
    args.student_num = 1
    log.info(f'args:{args}')

    if args.task_name == "sum":
        datamodule = ExtractiveSummaryDataModule(args)
        datamodule.prepare(args)
        num_labels = 1
    elif args.task_name in ["yahoo", "dbpedia", "agnews", "ag_news", "amazon_review", "yahoo_answers", "yelp_review"]:
        datamodule = SelfUdaDataModule(args)
        num_labels = datamodule.num_labels

    teacher_model = DynamicTeacher(model_name_or_path="bert-base-uncased",
                                   num_classes=num_labels,
                                   extractive_summary=args.task_name == "sum")
    ckpt = load_checkpoints(args.teacher_model, args.gpu if args.cuda else "cpu")
    teacher_model.load_state_dict(ckpt["model"])
    student_model = ClusterSum(config_name_or_path=args.config_name_or_path,
                               student_num=args.student_num,
                               fit_size=teacher_model.teacher.config.hidden_size,
                               num_classes=num_labels,
                               extractive_summary=args.task_name == "sum",
                               model_from_pretrained=args.model_from_pretrained)

    _, _, student_distill_optimizer, student_distill_scheduler, student_mutual_optimizer, student_mutual_scheduler = get_optimizer_and_scheduler(
        student_model, args, is_teacher=False)

    train_dataloader, unsupervised_dataloader = datamodule.train_dataloader(), datamodule.unsupervised_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()
    train_dataiter = iter(train_dataloader)
    unsupervised_dataiter = iter(unsupervised_dataloader)

    device = torch.device("cpu")
    if args.cuda:
        log.info(f"Use gpu: {args.gpu}")
        device = torch.device(args.gpu)
        student_model = student_model.to(torch.device(args.gpu))
        teacher_model = teacher_model.to(torch.device(args.gpu))

    loss_MSE = nn.MSELoss()
    if args.task_name == "sum":
        loss_CE = nn.BCEWithLogitsLoss()
    else:
        loss_CE = nn.CrossEntropyLoss()

    current_step = 0
    distill_loss_dic = {'total_loss': 0.0, 'sup_loss': 0.0, 'att_loss': 0.0, 'rep_loss': 0.0, 'logits_loss': 0.0}
    best_eval_loss = np.inf
    best_metric_score = np.NINF
    best_loss_checkpoints_filename = ""
    best_metric_checkpoints_filename = ""
    log.info("***** Running training *****")
    log.info("  Supervised Batch size = %d", train_dataloader.batch_size)
    log.info("  Unsupervised Batch size = %d", unsupervised_dataloader.batch_size)
    log.info("  Num steps = %d", args.num_training_steps)

    while current_step < args.num_training_steps:
        teacher_model.train()
        student_model.train()

        # log.info("1:{}".format(torch.cuda.memory_allocated(0)))

        try:
            batch = next(train_dataiter)
        except StopIteration:
            train_dataiter = iter(train_dataloader)
            batch = next(train_dataiter)
        except Exception as e:
            log.error(f"Error when loading data: {e}")
            log.error(traceback.format_exc())
            exit()

        try:
            unsupervised_batch = next(unsupervised_dataiter)
        except StopIteration:
            unsupervised_dataiter = iter(unsupervised_dataloader)
            unsupervised_batch = next(unsupervised_dataiter)
        except Exception as e:
            log.error(f"Error when loading data: {e}")
            log.error(traceback.format_exc())
            exit()

        batch, unsupervised_batch = batch.to(device), unsupervised_batch.to(device)

        # ***** Student distillation step *****
        _, student_encoder_layers, student_encoder_atts, _ = student_model(unsupervised_batch, distill_step=True)
        with torch.no_grad():
            teacher_logits, teacher_encoder_layers, teacher_encoder_atts = teacher_model(unsupervised_batch,
                                                                                         distill_step=True)
        teacher_layer_num = len(teacher_encoder_atts)
        student_layer_num = len(student_encoder_atts[0])
        assert teacher_layer_num % student_layer_num == 0
        layers_per_block = int(teacher_layer_num / student_layer_num)

        att_loss = 0.0
        new_teacher_atts = [teacher_encoder_atts[i * layers_per_block + layers_per_block - 1]
                            for i in range(student_layer_num)]
        for student_att, teacher_att in zip(student_encoder_atts[0], new_teacher_atts):
            att_loss += loss_MSE(student_att, teacher_att)

        rep_loss = 0.0
        new_teacher_reps = [teacher_encoder_layers[i * layers_per_block] for i in range(student_layer_num + 1)]
        for student_rep, teacher_rep in zip(student_encoder_layers[0], new_teacher_reps):
            rep_loss += loss_MSE(student_rep, teacher_rep)

        distill_loss = att_loss + rep_loss
        distill_loss_dic['total_loss'] += distill_loss.item()
        distill_loss_dic['att_loss'] += att_loss.item()
        distill_loss_dic['rep_loss'] += rep_loss.item()
        student_distill_optimizer[0].zero_grad()
        distill_loss.backward()
        student_distill_optimizer[0].step()
        student_distill_scheduler[0].step()

        unsupervised_student_logits, _, _, _ = student_model(unsupervised_batch, distill_step=False)

        supervised_student_logits, _, _, _ = student_model(batch, distill_step=False)

        sup_loss = loss_CE(supervised_student_logits[0], batch.labels)
        logits_loss = loss_MSE(unsupervised_student_logits[0], teacher_logits)

        prediction_loss = sup_loss + logits_loss

        distill_loss_dic['total_loss'] += distill_loss.item()
        distill_loss_dic['sup_loss'] += sup_loss.item()
        distill_loss_dic['logits_loss'] += logits_loss.item()
        student_mutual_optimizer[0].zero_grad()
        prediction_loss.backward()
        student_mutual_optimizer[0].step()
        student_mutual_scheduler[0].step()

        # log.info("4:{}".format(torch.cuda.memory_allocated(0)))

        current_step = current_step + 1

        if current_step % 100 == 0:
            log.info(f"Step {current_step:3d}")
            log.info(
                f"Student in distill stage | total loss {(distill_loss_dic['total_loss'] / 100.0):5.4f} | sup loss {(distill_loss_dic['sup_loss'] / 100.0):5.4f} | att loss {(distill_loss_dic['att_loss'] / 100.0):5.4f} | rep loss {(distill_loss_dic['rep_loss'] / 100.0):5.4f} | logits loss {(distill_loss_dic['logits_loss'] / 100.0):5.4f}")
            distill_loss_dic['total_loss'] = 0.0
            distill_loss_dic['att_loss'] = 0.0
            distill_loss_dic['rep_loss'] = 0.0
            distill_loss_dic['logits_loss'] = 0.0
            distill_loss_dic['sup_loss'] = 0.0

        if current_step % args.val_interval == 0:
            eval_loss, metric_result = evaluate(student_model,
                                                val_dataloader,
                                                0,
                                                args.extract_nsents,
                                                device,
                                                task_name=args.task_name,
                                                is_student=True,
                                                pyrouge=False,
                                                trigram_block=args.do_block)

            checkpoints = {
                "step": current_step,
                "model": student_model.state_dict(),
            }
            checkpoints_filename = os.path.join(args.root_dir, args.ckpt_dir, f"model_step_{current_step}.ckpt")
            save_checkpoints(checkpoints_filename, checkpoints)

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_loss_checkpoints_filename = checkpoints_filename
            tmp_metric_score = sum(metric_result.values())
            if tmp_metric_score > best_metric_score:
                best_metric_score = tmp_metric_score
                best_metric_checkpoints_filename = checkpoints_filename

    log.info("Train end.")
    log.info(f"The best loss checkpoint file is in {best_loss_checkpoints_filename}")
    log.info(f"The best rouge checkpoint file is in {best_metric_checkpoints_filename}")

    if args.test_after_train:
        ckpt = load_checkpoints(best_loss_checkpoints_filename, args.gpu if args.cuda else "cpu")
        student_model.load_state_dict(ckpt["model"])
        log.info("Test the best loss checkpoints.")
        evaluate(student_model,
                 test_dataloader,
                 0,
                 args.extract_nsents,
                 device,
                 task_name=args.task_name,
                 is_student=True,
                 pyrouge=True,
                 trigram_block=args.do_block)
        ckpt = load_checkpoints(best_rouge_checkpoints_filename, args.gpu if args.cuda else "cpu")
        student_model.load_state_dict(ckpt["model"])
        log.info("Test the best rouge checkpoints.")
        evaluate(student_model,
                 test_dataloader,
                 0,
                 args.extract_nsents,
                 device,
                 task_name=args.task_name,
                 is_student=True,
                 pyrouge=True,
                 trigram_block=args.do_block)

    return best_loss_checkpoints_filename, best_rouge_checkpoints_filename


if __name__ == "__main__":
    set_verbosity_error()
    args = parse_args()

    if args.seed > 0:
        log.info(f"Set seed to {args.seed}")
        seed_everything(args.seed)

    if args.do_cluster:
        log.info(f"Do cluster training!")
        best_loss_checkpoints_filename, best_rouge_checkpoints_filename = cluster_train(args)
    elif args.do_normal:
        log.info(f"Do normal training!")
        best_loss_checkpoints_filename, best_rouge_checkpoints_filename = normal_train(args)
    elif args.do_distill:
        log.info(f"Do distill training!")
        best_loss_checkpoints_filename, best_rouge_checkpoints_filename = distill_train(args)
