import torch
from torchmetrics import Metric


class SequenceAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False, padding_idx=-100):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        #preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        padding_indices = (target == -100)
        equal = target == preds
        equal[padding_indices] = True

        self.correct += torch.sum(torch.all(equal, dim=1))
        self.total += target.shape[0]

    def compute(self):
        return self.correct.float() / self.total
