from mlutils import Trainer
from torch import Tensor
from cvutils import transform as tf


class BaseTrainer(Trainer):
    def __init__(self, opt, device_id=None):
        super().__init__(opt, device_id=device_id)
        self.to_255 = tf.DeNormalize(
            mean=opt.image_mean,
            std=opt.image_std
        )

    def show_images(self, title: str, images: Tensor) -> None:
        images = self.to_255(images)
        self.dashboard.add_image(title, images, rgb=True)
