import math
import os
import torch
import wandb
from typing import Optional, Tuple, Union
from argparse import Namespace
from diffusers.models.unet_2d import UNet2DOutput
from torch.optim import *
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from torchvision.datasets import CIFAR10
from transformers import get_cosine_schedule_with_warmup

from ylcm.dataset import get_dataset
from ylcm.pipeline import ConsistencyPipeline
from ylcm.models import get_model
from ylcm.utils import make_grid
from ylcm.losses import get_loss_fn
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
import pytorch_lightning as pl
class Consistency(pl.LightningModule):
    def __init__(self,
                 config:Namespace) -> None:
        super(Consistency, self).__init__()
        self.config = config
        if config.push_to_hub:
            self.init_huggingface_hub(config)
        self.save_hyperparameters(config)
        self.model, self.ema = get_model(config.model)
        self.loss_fn = get_loss_fn(config.loss_fn)
        self.loss_metirc = MeanMetric()
        self.ema_decay_metric = MeanMetric()
        self.N_metric = MeanMetric()
        self.automatic_optimization=False
    def init_huggingface_hub(self, config):
        import huggingface_hub
        self.token = huggingface_hub.HfFolder.get_token()
        username = huggingface_hub.whoami(self.token)["name"]
        full_repo_name = f"{username}/{os.path.split(config.output_dir)[-1]}"

        huggingface_hub.create_repo(full_repo_name, exist_ok=True, token=self.token)

        self.repo = huggingface_hub.Repository(
            local_dir=self.config.output_dir,
            clone_from=full_repo_name,
            token=self.token,
        )
    def prepare_data(self) -> None:
        if self.config.dataset_name in ['CIFAR10CMDataset']:
            CIFAR10(".", train=True, download=True)
    def setup(self, stage: str) -> None:
        self.train_dataset =get_dataset(self.config)
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                                batch_size=self.config.batch_size,
                                shuffle=True,
                                num_workers=self.config.workers,
                                pin_memory=True,
                                drop_last=True)
    def configure_optimizers(self):
        self.optimizer = eval(self.config.optimizer)(self.model.parameters(), lr=self.config.learning_rate)
        lr_warmup_steps = len(self.train_dataloader()) * self.config.warm_epochs
        self.train_total_steps = len(self.train_dataloader()) * self.config.num_epochs
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=self.train_total_steps)
        return [self.optimizer], [self.lr_scheduler]

    def kerras_boundaries(self,
                          i : torch.Tensor,
                          rou : int,
                          eps : float,
                          N : int,
                          T : int
                          ) -> torch.Tensor:
        # This will be used to generate the boundaries for the time discretization

        return torch.tensor(
            (eps ** (1 / rou) + i / (N - 1) * (T ** (1 / rou) - eps ** (1 / rou)))
            ** rou
            )
    def forward(
        self,
        model: nn.Module,
        samples: torch.Tensor,
        times: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        clip: bool = True,
    ):
        return self._forward(self.ema if self.config.use_ema else model, samples, times, class_labels, clip)
    def _forward(
        self,
        model: nn.Module,
        samples: torch.Tensor,
        times: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        clip: bool = True,
    ):
        skip_coef = self.config.data_std**2 / (
            (times - self.config.eps).pow(2) + self.config.data_std**2
        )
        out_coef = self.config.data_std * (times - self.config.eps) / (times.pow(2) + self.config.data_std**2).pow(0.5)

        out: UNet2DOutput = model(samples, times, class_labels)

        out = samples * skip_coef[:, None, None, None] + out.sample * out_coef[:, None, None, None]

        if clip:
            return out.clamp(-1.0, 1.0)

        return out
    @torch.no_grad()
    def ema_update(self, N:int) -> None:
        param = [p.data for p in self.model.parameters()]
        param_ema = [p.data for p in self.ema.parameters()]

        torch._foreach_mul_(param_ema, self.ema_decay)
        torch._foreach_add_(param_ema, param, alpha=1 - self.ema_decay)
        self.ema_decay_metric(self.ema_decay)
        # if self.config.use_wandb:
        #     wandb.log({'ema_decay' : self.ema_decay(N)})
        self.log(
            "ema_decay",
            self.ema_decay_metric,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
    @property
    def ema_decay(self) -> float:
        return math.exp(self.config.s0 * math.log(self.config.mu0) / self.N)
    @property
    def N(self) -> int:
        return  math.ceil(math.sqrt((self.trainer.global_step * ((
                          self.config.s1 + 1) ** 2 - self.config.s0 ** 2) /
                          self.trainer.estimated_stepping_batches) + self.config.s0 ** 2) - 1) + 1
    def loss(self,
             x:torch.Tensor, #[b, c, h, w]
             z:torch.Tensor, #[b, c, h, w]
             t1:torch.Tensor, #[b]
             t2:torch.Tensor, #[b]
             l:Optional[torch.Tensor]=None
             ) -> torch.Tensor:
        x2 = x + z * t2[:, None, None, None]
        x2 = self._forward(self.model, x2, t2, l)

        with torch.no_grad():
            x1 = x + z * t1[:, None, None, None]
            x1 = self._forward(self.ema, x1, t1, l)
        return self.loss_fn(x1, x2)

    def save_samples(self, epoch:int) -> None:
        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        if self.config.conditional:
            images = self.pipeline(
                batch_size=self.config.num_samples,
                generator=torch.manual_seed(self.config.seed),
                num_class=self.config.nc,
                eps = self.config.eps,
                T = self.config.T,
                data_std = self.config.data_std,
                num_inference_steps = self.config.sample_steps
            ).images
        else:
            images = self.pipeline(
                batch_size=self.config.num_samples,
                generator=torch.manual_seed(self.config.seed),
                eps=self.config.eps,
                T=self.config.T,
                data_std=self.config.data_std,
                num_inference_steps=self.config.sample_steps
            ).images
        bn = len(images)
        rows = int(math.sqrt(bn))
        while (bn % rows != 0):
            rows -= 1
        cols = bn // rows
        # Make a grid out of the images
        image_grid = make_grid(images, rows=rows, cols=cols)

        # Save the images
        test_dir = os.path.join(self.config.exp, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")
        # save_image(
        #     image_grid,
        #     f"{test_dir}/{epoch:04d}.png",
        #     "png",
        # )
        if self.config.use_wandb:
            image_grid = wandb.Image(image_grid, caption=f'Epoch {epoch}')
            wandb.log({'sample_images':image_grid})
        del images, image_grid
    def training_step(self,
                      batch:Union[torch.Tensor, Tuple],
                      batch_idx:int
                      ) -> torch.Tensor:
        if self.config.conditional:
            x, l = batch
        else:
            x = batch
            l = None
        self.N_metric(self.N)
        self.log(
            "N",
            self.N_metric,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            logger=True if self.config.use_wandb else False
        )
        device = x.device
        optimizer = self.optimizers()
        lr_scheduler = self.lr_schedulers()
        z = torch.randn_like(x)
        t = torch.randint(0, self.N - 1, (x.shape[0],), device=device)
        t_0 = self.kerras_boundaries(t, self.config.rho, self.config.eps, self.N, self.config.T).to(device)[t]
        t_1 = self.kerras_boundaries(t+1, self.config.rho, self.config.eps, self.N, self.config.T).to(device)[t]
        loss = self.loss(x, z, t_0, t_1, l)
        optimizer.zero_grad()
        self.manual_backward(loss)
        self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        # 调用优化器根据梯度更新模型参数
        optimizer.step()
        self.ema_update(self.N)
        # 更新学习率，每训练一个批次学习率都会变化，学习率先会经过热身阶段从很小的值变成初始设置的值然后学习率会不断下降
        lr_scheduler.step()
        self.loss_metirc(loss)
        self.log(
            "loss",
            self.loss_metirc,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            logger=True if self.config.use_wandb else False
        )
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        self.log(
            "men",
            mem,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            logger=True if self.config.use_wandb else False,
        )
        self.log(
            "lr",
            lr_scheduler.get_last_lr()[0],
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            logger=True if self.config.use_wandb else False
        )
        return loss

    def on_train_epoch_end(self) -> None:

        if (
            ((self.trainer.current_epoch + 1) % self.config.save_image_epochs == 0)
            or self.trainer.current_epoch + 1 == self.config.num_epochs
        ):
            self.pipeline = ConsistencyPipeline(
                unet=self.ema if self.config.use_ema
                else self.model)
            self.save_samples(self.trainer.current_epoch)
        if (
            ((self.trainer.current_epoch + 1) % self.config.save_model_epochs == 0)
            or self.trainer.current_epoch + 1 == self.config.num_epochs
        ):
            self.pipeline.save_pretrained(self.config.exp)
            print('🚀consistency model pipeline has saved!')
            if self.trainer.current_epoch + 1 == self.config.num_epochs:
                if self.config.push_to_hub:
                    self.repo.push_to_hub(
                        commit_message=f"Epoch {self.trainer.current_epoch + 1}",
                        blocking=False,
                    )
        torch.cuda.empty_cache()


