from ..base import BaseTrainer


class ViTBaseTrainer(BaseTrainer):
    def __init__(self, opt, device_id=None):
        super().__init__(opt, device_id=device_id)
