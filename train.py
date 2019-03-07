import gc
from collections import namedtuple
from datetime import datetime
from pprint import pprint

import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import (
    Events, create_supervised_evaluator, create_supervised_trainer
)
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Accuracy, Loss, RunningAverage, Precision, Recall
from PIL import Image
from torch import nn
# from pretrainedmodels.models import bninception
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.models import inception_v3

from models.model import count_params
from putils.data import load_data
from putils.generator import CancerDatset, RandomSamplerOk
from putils.handlers import SummaryWriter
from putils.metrics import ConfusionMatrix, Mean

_Data = namedtuple('_Data', ['slide_path', 'coord', 'label'])


def main(folder_with_data, patch_size, zoom, batch_size, epochs):
    train_data, valid_data = load_data(folder_with_data)
    transforms = T.Compose(
        [
            T.RandomVerticalFlip(),
            T.RandomHorizontalFlip(),
            T.RandomRotation(90, resample=Image.BILINEAR),
        ]
    )
    trainDataset = CancerDatset(
        train_data, patch_size, zoom, transforms, noise=True
    )
    validDataset = CancerDatset(valid_data, patch_size, zoom, noise=False)
    trainDataloader = DataLoader(
        trainDataset,
        sampler=RandomSamplerOk(
            trainDataset,
            num_samples=len(trainDataset) // 10,
            replacement=True,
        ),
        batch_size=batch_size,
        num_workers=6
    )
    validDataloader = DataLoader(
        validDataset,
        sampler=RandomSamplerOk(
            validDataset,
            num_samples=len(validDataset) // 10,
            replacement=True,
        ),
        batch_size=batch_size,
        num_workers=6,
    )
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = inception_v3(num_classes=1, aux_logits=False)
    print('Total params:', count_params(model))
    optimizer = Adam(model.parameters(), lr=5e-4)
    criterion = nn.BCEWithLogitsLoss()

    tag = datetime.now().strftime('%m-%d--%H-%M')
    t_writer = SummaryWriter(model, f'./logs/{tag}/train', (3, 256, 256))
    v_writer = SummaryWriter(model, f'./logs/{tag}/val', (3, 256, 256))

    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device
    )
    # output_transform = lambda t: (t[0].sigmoid(), t[1])
    metrics = {'cm': Mean(ConfusionMatrix(2)), 'loss': Loss(criterion)}
    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device
    )

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    ProgressBar(persist=True).attach(trainer, ['loss'])
    ProgressBar(persist=True).attach(evaluator)

    @trainer.on(Events.EPOCH_COMPLETED)
    def validate(engine):
        # pylint: disable=no-member
        for cat, (loader, writer) in dict(
            train=(trainDataloader, t_writer), val=(validDataloader, v_writer)
        ).items():
            ms = evaluator.run(loader).metrics  # run evaluation

            try:
                v = ms.pop('cm').cpu()
                # pprint(v.numpy().round(6).tolist())
            except KeyError:
                raise NotImplementedError('confusion matrix must be')
            d = dict(
                acc=v.diag().sum(),
                recall=v[1, 1] / (v[:, 1].sum() + 1e-7),
                precision=v[1, 1] / (v[1, :].sum() + 1e-7),
                f1=2 * v[1, 1] / (v[1, :].sum() + v[:, 1].sum() + 1e-7),
                kappa=1 - (1 - v.diag().sum()) / (1 - v.sum(0) @ v.sum(1))
            )
            ms.update({k: v_.item() for k, v_ in d.items()})

            for k, v in ms.items():
                if not isinstance(v, torch.Tensor):
                    writer.add_scalar(k, v, engine.state.epoch)

            ms = {f'{cat}_{k}': v_ for k, v_ in ms.items()}
            engine.state.metrics.update(ms)

        print(f'[Epoch {engine.state.epoch}]:')
        pprint(engine.state.metrics)

    now = datetime.now()
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        ModelCheckpoint(
            f'./checkpoints/{now.year%100:02d}-{now.month:02d}-{now.day:02d}',
            'model',
            n_saved=3,
            score_function=lambda e: e.state.metrics['val_acc'],
            score_name='acc',
            create_dir=True,
            require_empty=False,
            save_as_state_dict=True,
        ), {'inception': model}
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda *_: (print(datetime.now().strftime('%H:%M:%S'), end='\n\n'),
                    torch.cuda.empty_cache(),
                    gc.collect())
    )
    trainer.run(trainDataloader, max_epochs=epochs)


if __name__ == '__main__':
    main('./data/pickle/', (224, 224), 0, 12, 3000)

    # valid = load_data('./data/pickle/')
    # print(valid)