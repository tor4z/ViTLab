import torch
from torch import Tensor
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from cfg import Opts
from mlutils import mod, gen
from vitlab.network.vit.vit import VisualTransformerCls

from .base import ViTBaseTrainer


__all__ = ['ViTTrainer']


@mod.register('arch')
class ViTTrainer(ViTBaseTrainer):
    @gen.synchrony
    def __init__(self, opt: Opts) -> None:
        super().__init__(opt)
        net = VisualTransformerCls(opt)
        self.optimizer = SGD(
            net.parameters(), lr=opt.lr, momentum=0.9,
            weight_decay=opt.get('weight_decay', 1.0e-3))
        self.scheduler = StepLR(self.optimizer, 4, 0.9)
        self.net = yield self.to_gpu(net)
        self.loss_fn = nn.CrossEntropyLoss()

    @gen.detach_cpu
    @gen.synchrony
    def train_step(self, item):
        images, labels = item
        images = yield self.to_gpu(images)
        labels = yield self.to_gpu(labels)
        labels = labels.type(torch.int64)

        self.optimizer.zero_grad()
        logits = self.net(images)
        loss = self.loss_fn(logits, labels)
        loss.backward()
        self.optimizer.step()

        preds = self.logit_to_preds(logits)
        return loss, preds, labels

    @gen.detach_cpu
    @gen.synchrony
    def eval_step(self, item):
        images, labels = item
        images = yield self.to_gpu(images)
        labels = yield self.to_gpu(labels)
        labels = labels.type(torch.int64)

        logits = self.net(images)
        loss = self.loss_fn(logits, labels)

        self.show_images('eval_image', images)
        preds = self.logit_to_preds(logits)
        return loss, preds, labels

    @gen.synchrony
    def inference(self, inp: Tensor) -> Tensor:
        inp = yield self.to_gpu(inp)

        if inp.ndim == 3:
            inp = inp.unsqueeze(0)

        with torch.no_grad():
            logits = self.net(inp)

        self.show_images('inference_image', inp)
        preds = self.logit_to_preds(logits)
        preds = yield self.to_cpu(preds.detach())
        return preds

    def on_epoch_end(self) -> None:
        self.scheduler.step()
