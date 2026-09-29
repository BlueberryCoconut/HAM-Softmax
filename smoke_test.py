import argparse
import gc
import os

import torch
from torch.utils.data import DataLoader

from datasets import BaseDataset
from loss import AMSoftmax, HyperbolicAMSoftmax, Softmax
from model import ECAPA_TDNN
from utils import get_data_path


def run_case(name, loss_factory, data, labels, n_class):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    encoder = ECAPA_TDNN(C=1024).cuda().train()
    criterion = loss_factory(n_class).cuda().train()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(criterion.parameters()), lr=1e-3
    )

    optimizer.zero_grad(set_to_none=True)
    embeddings = encoder(data.cuda(non_blocking=True), aug=True)
    result = criterion(embeddings, labels.cuda(non_blocking=True))
    loss = result[0].mean()
    loss.backward()
    optimizer.step()
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f'{name}: loss={loss.item():.6f}, peak_cuda_gb={peak_gb:.3f}')

    del optimizer, criterion, encoder, embeddings, result, loss
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    train_list = os.path.join(os.path.dirname(__file__), 'data', 'v1_clean.txt')
    _, train_path = get_data_path('v1')
    dataset = BaseDataset(
        train_list=train_list, train_path=train_path, num_frames=200, augment=False
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    _, data, labels = next(iter(loader))
    n_class = dataset.get_speaker_number()

    cases = [
        ('Softmax', lambda n: Softmax(n)),
        ('H-Softmax', lambda n: HyperbolicAMSoftmax(n, m=0.0, s=30, c=5)),
        ('AM-Softmax', lambda n: AMSoftmax(n, m=0.2, s=30)),
        ('HAM-Softmax', lambda n: HyperbolicAMSoftmax(n, m=0.2, s=30, c=3)),
    ]
    for name, factory in cases:
        run_case(name, factory, data, labels, n_class)


if __name__ == '__main__':
    main()
