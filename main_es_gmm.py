"""ES-GMM training for speaker verification with noisy labels.

Implements the baseline described in Yao et al., ICASSP 2025: after a
warm-up period, fit a two-component GMM to the cosine similarities between
utterance embeddings and their labelled-speaker centres.  The component with
the larger mean similarity is retained for the next epoch.
"""

import argparse
import collections
import datetime
import glob
import os
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from datasets import BaseDataset
from loss import HyperbolicAMSoftmax
from model import ECAPA_TDNN
from utils import get_data_path, get_evaluator, get_loss_function, get_optimizer, get_scheduler
from tools import Logger


parser = argparse.ArgumentParser(description="ES-GMM ECAPA trainer")
parser.add_argument("--save_root", type=str, default="exps")
parser.add_argument("--num_frames", type=int, default=200)
parser.add_argument("--max_epoch", type=int, default=100)
parser.add_argument("--test_step", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_cpu", type=int, default=0)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_decay", type=float, default=0.97)
parser.add_argument("--dataset", type=str, default="v1")
parser.add_argument("--train_list", type=str, required=True)
parser.add_argument("--augment", action="store_true")
parser.add_argument("--C", type=int, default=1024)
parser.add_argument("--m", type=float, default=0.2)
parser.add_argument("--s", type=float, default=30)
parser.add_argument("--lambda_1", type=float, default=1.0)
parser.add_argument("--lambda_2", type=float, default=0.0)
parser.add_argument("--loss_type", type=str, choices=["am", "aam"], default="am",
                    help="Euclidean angular-margin loss used alongside ES-GMM")
parser.add_argument("--h_C", type=float, default=3.0)
parser.add_argument("--h_m", type=float, default=0.2)
parser.add_argument("--h_s", type=float, default=30.0)
parser.add_argument("--warmup_epochs", type=int, default=5)
parser.add_argument("--gmm_reg_covar", type=float, default=1e-6)
parser.add_argument("--selector_batch_size", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--postfix", type=str, required=True)


class Model(nn.Module):
    def __init__(self, C, n_class, m, s, lr, lr_decay, h_C, h_m, h_s, loss_type="am"):
        super().__init__()
        self.speaker_encoder = ECAPA_TDNN(C=C).cuda()
        self.speaker_loss1 = get_loss_function(loss_type, n_class, m, s)
        self.speaker_loss2 = HyperbolicAMSoftmax(n_class=n_class, m=h_m, s=h_s, c=h_C).cuda()
        self.optim = get_optimizer(self, lr)
        self.scheduler = get_scheduler(self.optim, lr_decay)

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        state = self.state_dict()
        loaded = torch.load(path, map_location="cpu")
        for name, param in loaded.items():
            target = name if name in state else name.replace("module.", "")
            if target not in state or state[target].size() != param.size():
                continue
            state[target].copy_(param)


def build_paths(args):
    if args.lambda_2 > 0:
        experiment = "v1_ham_esgmm_lam1={}_lam2={}_C={}_m={}_s={}".format(
            args.lambda_1, args.lambda_2, int(args.h_C), args.h_m, args.h_s)
    else:
        experiment = "v1_esgmm_{}_m={}_s={}".format(args.loss_type, args.m, args.s)
    if args.augment:
        experiment += "_augment"
    experiment += "_" + args.postfix
    args.save_path = os.path.join(args.save_root, experiment)
    args.model_save_path = os.path.join(args.save_path, "model")
    args.log_path = os.path.join(args.save_path, "train.log")
    args.score_save_path = os.path.join(args.save_path, "result_{}.csv".format(experiment))
    os.makedirs(args.model_save_path, exist_ok=True)


@torch.no_grad()
def select_clean_indices(model, dataset, args, logger):
    """Return indices belonging to the high-similarity GMM component."""
    loader = DataLoader(
        dataset,
        batch_size=args.selector_batch_size,
        shuffle=False,
        num_workers=args.n_cpu,
        drop_last=False,
    )
    model.eval()
    all_indices, all_labels, all_embeddings = [], [], []
    for batch in tqdm(loader, desc="ES-GMM select", ncols=120, mininterval=10):
        if args.augment:
            indices, data, _, labels = batch
        else:
            indices, data, labels = batch
        embeddings = model.speaker_encoder.forward(data.cuda(), aug=False)
        all_indices.append(indices.numpy())
        all_labels.append(labels.numpy())
        all_embeddings.append(F.normalize(embeddings, p=2, dim=1).cpu().numpy())

    indices = np.concatenate(all_indices)
    labels = np.concatenate(all_labels)
    embeddings = np.concatenate(all_embeddings).astype(np.float32, copy=False)
    centres = np.zeros((args.n_class, embeddings.shape[1]), dtype=np.float32)
    np.add.at(centres, labels, embeddings)
    counts = np.bincount(labels, minlength=centres.shape[0]).astype(np.float32)
    centres /= np.maximum(counts[:, None], 1.0)
    centres /= np.maximum(np.linalg.norm(centres, axis=1, keepdims=True), 1e-12)
    similarities = np.sum(embeddings * centres[labels], axis=1)

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        reg_covar=args.gmm_reg_covar,
        random_state=args.seed,
        n_init=1,
    )
    components = gmm.fit_predict(similarities.reshape(-1, 1))
    clean_component = int(np.argmax(gmm.means_.reshape(-1)))
    keep = components == clean_component
    clean_indices = indices[keep].tolist()
    logger.print(
        "ES-GMM: means={}, weights={}, retained={}/{} ({:.2f}%), similarity=[{:.4f}, {:.4f}]".format(
            np.round(gmm.means_.reshape(-1), 6).tolist(),
            np.round(gmm.weights_, 6).tolist(),
            len(clean_indices), len(indices), 100.0 * len(clean_indices) / len(indices),
            float(similarities.min()), float(similarities.max()),
        )
    )
    return clean_indices


def train_epoch(model, epoch, dataset, selected_indices, args):
    model.train()
    model.scheduler.step(epoch - 1)
    loader_dataset = dataset if selected_indices is None else Subset(dataset, selected_indices)
    loader = DataLoader(loader_dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.n_cpu, drop_last=True)
    loss_total, accuracy_total, steps = 0.0, 0.0, 0
    progress = tqdm(loader, desc="train", ncols=120, mininterval=10)
    for batch in progress:
        if args.augment:
            _, data, aug_data, labels = batch
        else:
            _, data, labels = batch
        labels = labels.cuda()
        embeddings = model.speaker_encoder.forward(data.cuda(), aug=True)
        if args.lambda_1 > 0:
            loss1, _, accuracy1 = model.speaker_loss1.forward(embeddings, labels)
            loss1 = loss1.mean()
        else:
            loss1, accuracy1 = torch.zeros((), device=labels.device), 0.0
        if args.lambda_2 > 0:
            loss2, _, accuracy2, _ = model.speaker_loss2.forward(embeddings, labels)
            loss2 = loss2.mean()
        else:
            loss2, accuracy2 = torch.zeros((), device=labels.device), 0.0
        if args.augment:
            aug_embeddings = model.speaker_encoder.forward(aug_data.cuda(), aug=True)
            if args.lambda_1 > 0:
                aug_loss1, _, _ = model.speaker_loss1.forward(aug_embeddings, labels)
                loss1 = loss1 + aug_loss1.mean()
            if args.lambda_2 > 0:
                aug_loss2, _, _, _ = model.speaker_loss2.forward(aug_embeddings, labels)
                loss2 = loss2 + aug_loss2.mean()
        loss = args.lambda_1 * loss1 + args.lambda_2 * loss2
        model.zero_grad()
        loss.backward()
        model.optim.step()
        steps += 1
        loss_total += loss.item()
        accuracy_total += accuracy2 if args.lambda_2 > 0 else accuracy1
        progress.set_postfix(loss="{:.4f}".format(loss_total / steps),
                             acc="{:.2f}".format(100 * accuracy_total / steps))
    if steps == 0:
        raise RuntimeError("ES-GMM selected too few samples for one training batch")
    return loss_total / steps, 100 * accuracy_total / steps, model.optim.param_groups[0]["lr"]


def make_model(args):
    return Model(args.C, args.n_class, args.m, args.s, args.lr, args.lr_decay,
                 args.h_C, args.h_m, args.h_s, args.loss_type)


def average_final_model(args, model):
    final_path = os.path.join(args.model_save_path, "model_final.model")
    models = []
    for epoch in range(args.max_epoch - 4, args.max_epoch + 1):
        item = make_model(args)
        item.load_parameters(os.path.join(args.model_save_path, "model_{:04d}.model".format(epoch)))
        models.append(item)
    states = [item.state_dict() for item in models]
    averaged = collections.OrderedDict()
    for key in states[0]:
        averaged[key] = sum(state[key] for state in states) / len(states)
    model.load_state_dict(averaged)
    model.save_parameters(final_path)
    return model


def main(args):
    build_paths(args)
    logger = Logger(args.log_path)
    logger.print("ES-GMM start: {}".format(datetime.datetime.now()))
    logger.print("train_list={}, warmup_epochs={}, augment={}".format(
        args.train_list, args.warmup_epochs, args.augment))
    _, args.train_path = get_data_path(args.dataset)
    dataset = BaseDataset(train_list=args.train_list, train_path=args.train_path,
                          num_frames=args.num_frames, augment=args.augment)
    args.n_class = int(dataset.get_speaker_number())
    evaluator = get_evaluator(args.dataset)
    files = sorted(glob.glob(os.path.join(args.model_save_path, "model_0*.model")))
    if files:
        epoch = int(os.path.splitext(os.path.basename(files[-1]))[0][6:]) + 1
        model = make_model(args)
        model.load_parameters(files[-1])
        logger.print("resumed from {}".format(files[-1]))
    else:
        epoch = 1
        model = make_model(args)
    selected_indices = None
    eers = [100.0]
    if not os.path.exists(args.score_save_path):
        with open(args.score_save_path, "w") as output:
            output.write("Time,Epoch,LR,Loss,Acc,Retained,EER,minDCF(0.05),bestEER\n")

    while epoch <= args.max_epoch:
        logger.print("Epoch {}".format(epoch))
        started = datetime.datetime.now()
        loss, accuracy, lr = train_epoch(model, epoch, dataset, selected_indices, args)
        model.save_parameters(os.path.join(args.model_save_path, "model_{:04d}.model".format(epoch)))
        retained = len(dataset) if selected_indices is None else len(selected_indices)
        eer = mindcf = None
        if epoch % args.test_step == 0:
            eer, mindcf = evaluator.eval(model)
            eers.append(eer)
            logger.print("ACC: {:.3f}, EER: {:.3f}, minDCF: {:.3f}, bestEER: {:.3f}".format(
                accuracy, eer, mindcf, min(eers)))
        with open(args.score_save_path, "a") as output:
            output.write("{},{},{},{},{},{},{},{},{}\n".format(
                datetime.datetime.now(), epoch, lr, loss, accuracy, retained, eer, mindcf, min(eers)))
        logger.print("epoch time: {}".format(datetime.datetime.now() - started))
        if epoch >= args.warmup_epochs and epoch < args.max_epoch:
            selected_indices = select_clean_indices(model, dataset, args, logger)
        epoch += 1

    final_model = average_final_model(args, model)
    logger.print("start final evaluation")
    eer, mindcf = evaluator.eval(final_model)
    logger.print("final, EER: {:.3f}, minDCF(0.05): {:.3f}, bestEER: {:.3f}".format(
        eer, mindcf, min(eers + [eer])))


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    time.sleep(2)
    main(args)
