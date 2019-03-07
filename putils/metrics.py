import torch
from dataclasses import dataclass
from ignite.metrics import Metric
from ignite.engine import Events
from ignite.exceptions import NotComputableError

# class ConfusionMatrix:
#     def __init__(self, output, nb_classes=2):
#         self.output = output
#         self.nb_classes = nb_classes

#     def __call__(self):
#         y_pred, y_true, *_ = self.output
#         y_pred = y_pred.round().long()
#         y_pred = torch.zeros(y_pred.shape[0],
#                              self.nb_classes).scatter_(1, y_pred, 1)
#         y_true = torch.t(
#             torch.zeros(y_true.shape[0],
#                         self.nb_classes).scatter_(1, y_true, 1)
#         )
#         cm = (y_true @ y_pred).double()

#         return cm / cm.sum([0, 1])


class Mean(Metric):
    def reset(self):
        self._value = None
        self._count = 0

    def update(self, output):
        self._count += 1
        if self._value is None:
            self._value = output.clone().cpu()
        else:
            self._value += (output.cpu() - self._value) / self._count

    def compute(self):
        if not self._count:
            raise NotComputableError(
                "Loss must have at least one example before it can be computed"
            )
        if self._value.numel() == 1:
            return self._value.item()
        return self._value

    def attach(self, engine, name):
        engine.add_event_handler(Events.EPOCH_STARTED, self.started)
        engine.add_event_handler(
            Events.ITERATION_COMPLETED, self.iteration_completed
        )
        engine.add_event_handler(
            Events.ITERATION_COMPLETED, self.completed, name
        )


@dataclass
class ConfusionMatrix:
    classes: int

    def __call__(self, output):
        y_pred, y_true, *_ = output
        y_pred = y_pred.sigmoid().round().long()
        y_true = y_true.long()

        place = torch.zeros(
            y_pred.shape[0],
            self.classes,
            device=y_pred.device,
            dtype=torch.double
        )
        y_pred = place.scatter(1, y_pred, 1)
        y_true = place.scatter(1, y_true, 1)

        cm = y_true.t() @ y_pred
        return cm / cm.sum()
