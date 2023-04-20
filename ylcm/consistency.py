import math
import os
import pickle
from time import time
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import wandb
import datetime

from diffusers.models.unet_2d import UNet2DOutput
from torch.optim import *
from torch import nn
from accelerate import Accelerator
from diffusers import UNet2DModel
from tqdm.auto import tqdm
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from transformers import get_cosine_schedule_with_warmup
from ylcm.config import get_dataloader
from ylcm.pipeline import ConsistencyPipeline
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torch.nn import MSELoss, L1Loss

class Consistency:
    def __init__(self, config=None):
        if config:
            self.init_base(config)
            self.init_dataloader(config)
            self.init_train_config(config)
            if config.resume_ckpt_path:
                self.init_resume(config)
            if config.use_wandb:
                self.init_wandb(config)
            if config.push_to_hub:
                self.init_huggingface_hub(config)
    def init_base(self, config):
        self.config = config
        if config.conditional:
            self.index2label = config.index2label
        self.logger = dict(
            train_noise_loss=0, train_total_num=0, train_loss_list=[]
        )
        self.init_data = False
        self.train_global_step = 0
        self.start_epoch = 1
    def init_dataloader(self, config):
        get_dataloader(config)
        self.train_dataloader = config.train_dataloader
        self.init_data=True
    def init_train_config(self, config):
        self.model = deepcopy(config.model)
        self.ema = deepcopy(self.model)
        self.get_optimizer(config)
        lr_warmup_steps = len(self.train_dataloader) * config.warm_epochs
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=len(self.train_dataloader) * config.num_epochs)
        self.accelerator = Accelerator(mixed_precision=config.mixed_precision)
        self.device = self.accelerator.device
        self.model = self.model.to(self.device)
        self.model, self.ema, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model,
            self.ema,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler)
        self.model_name = self.config.project_name
        self.loss_fn = eval(self.config.loss_fn).to(self.device)
    def init_resume(self, config):
        ckpt = torch.load(config.resume_ckpt_path)
        self.start_epoch = ckpt['epoch']
        self.logger = ckpt['logger']
        self.train_global_step = ckpt['global_steps']
        self.model.load_state_dict(ckpt['model'])
        self.ema.load_state_dict(ckpt['ema'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        if config.use_wandb:
            self.run_id = ckpt['run_id']
        self.config.output_dir = os.path.abspath(os.path.join(config.resume_ckpt_path, ".."))
        del ckpt
    def init_wandb(self, config):
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if config.resume_ckpt_path:
            self.run = wandb.init(project=config.project_name, id=self.run_id, resume='must')
        else:
            self.run = wandb.init(project=config.project_name, config=config.wandb_dict, name=nowtime, tags=[os.path.join(config.output_dir, self.model_name)])
            self.run_id = wandb.run.id
        # define our custom x axis metric
        wandb.define_metric("epoch")
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
    def get_optimizer(self, config):
        if isinstance(config.optimizer, SGD):
            self.optimizer = eval(config.optimizer)(self.model.parameters(), lr=config.learning_rate, momentum=config.momentum,weight_decay=config.weight_decay)
        else:
            self.optimizer = eval(config.optimizer)(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    def kerras_boundaries(self, rou, eps, N, T):
        # This will be used to generate the boundaries for the time discretization

        return torch.tensor(
            [
                (eps ** (1 / rou) + i / (N - 1) * (T ** (1 / rou) - eps ** (1 / rou)))
                ** rou
                for i in range(N)
            ]
        )
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
    def ema_update(self, N):
        param = [p.data for p in self.model.parameters()]
        param_ema = [p.data for p in self.ema.parameters()]

        torch._foreach_mul_(param_ema, self.ema_decay(N))
        torch._foreach_add_(param_ema, param, alpha=1 - self.ema_decay(N))
        if self.config.use_wandb:
            wandb.log({'ema_decay' : self.ema_decay(N)})

    def ema_decay(self, N):
        return math.exp(self.config.s0 * math.log(self.config.mu0) / N)
    def loss(self, x, z, t1, t2, l=None):
        x2 = x + z * t2[:, None, None, None]
        x2 = self._forward(self.model, x2, t2, l)

        with torch.no_grad():
            x1 = x + z * t1[:, None, None, None]
            x1 = self._forward(self.ema, x1, t1, l)

        return self.loss_fn(x1, x2)

    def make_grid(self, images, rows, cols):
        w, h = images[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        for i, image in enumerate(images):
            grid.paste(image, box=(i % cols * w, i // cols * h))
        return grid

    def save_samples(self, epoch):
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
        image_grid = self.make_grid(images, rows=rows, cols=cols)

        # Save the images
        test_dir = os.path.join(self.config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")
        if self.config.use_wandb:
            image_grid = wandb.Image(image_grid, caption=f'Epoch {epoch}')
            wandb.log({'sample_images':image_grid})

    def train(self):
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
        self.model.train()
        print("🚀yl-consistency model training starts!")
        t1 = time()
        for epoch in range(self.start_epoch, self.config.num_epochs+1):
            self.logger['train_noise_loss'] = 0
            self.logger['train_total_num'] = 0
            N = math.ceil(math.sqrt((epoch * ((self.config.s1 + 1) ** 2 - self.config.s0 ** 2) / self.config.num_epochs) + self.config.s0 ** 2) - 1) + 1
            boundaries = self.kerras_boundaries(self.config.rou, self.config.eps, N, self.config.T).to(self.device)
            with tqdm(total = len(self.train_dataloader), desc=f'train : Epoch [{epoch}/{self.config.num_epochs}]', postfix=dict,mininterval=0.3) as pbar:
                for datas in self.train_dataloader:
                    if self.config.conditional:
                        x, l = datas
                        x = x.to(self.device)
                        l = l.to(self.device)
                    else:
                        x = datas
                        x = x.to(self.device)
                        l = None
                    z = torch.randn_like(x)
                    t = torch.randint(0, N - 1, (x.shape[0],), device=self.device)
                    t_0 = boundaries[t]
                    t_1 = boundaries[t + 1]
                    loss = self.loss(x, z, t_0, t_1, l)
                    self.logger['train_noise_loss'] += loss.item()
                    self.logger['train_total_num'] += x.shape[0]
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    #反向传计算据梯度
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    #调用优化器根据梯度更新模型参数
                    self.optimizer.step()
                    self.ema_update(N)
                    #更新学习率，每训练一个批次学习率都会变化，学习率先会经过热身阶段从很小的值变成初始设置的值然后学习率会不断下降
                    self.lr_scheduler.step()
                    #每更新完一个批次都要把梯度清零，不然会累加起来
                    self.optimizer.zero_grad()
                    logs = {'mem' : mem,
                            "loss": loss.detach().item(),
                            "lr": self.lr_scheduler.get_last_lr()[0],
                            "step": self.train_global_step}
                    pbar.set_postfix(**logs)
                    pbar.update(1)
                    self.train_global_step += 1
                    wandb.log({"loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0], 'epoch':epoch}) if self.config.use_wandb else None
            self.pipeline = ConsistencyPipeline(unet=self.accelerator.unwrap_model(self.ema) if self.config.use_ema else self.accelerator.unwrap_model(self.model))
            if epoch % self.config.save_image_epochs == 0 or epoch == self.config.num_epochs:
                self.save_samples(epoch)
            if epoch % self.config.save_model_epochs == 0 or epoch == self.config.num_epochs:
                # self.ddpmpipeline.save_pretrained(self.config.output_dir)
                self.pipeline.save_pretrained(self.config.output_dir)
                if self.config.push_to_hub:
                    self.repo.push_to_hub(
                        commit_message=f"Epoch {epoch}",
                        blocking=False,
                    )
                print('🚀yl-consistency model has saved!')
            self.save_model(epoch, 'last')
            train_mean_loss = self.logger['train_noise_loss'] / self.logger['train_total_num']
            self.logger['train_loss_list'].append(train_mean_loss)
            print(f'train Epoch [{epoch}/{self.config.num_epochs}] loss:{train_mean_loss} ')
            if self.config.use_wandb:
                # define which metrics will be plotted against it
                wandb.define_metric("train_loss", step_metric="epoch")
                wandb.log({'epoch':epoch, 'train_loss':loss}) if self.config.use_wandb else None
        t2 = time()
        self.save_config()
        self.save_loss_csv()
        self.save_loss_curve()
        if self.config.use_wandb:
            if self.config.dataset_name in ['CIFAR10CMDataset']:
                self.arti_dataset = wandb.Artifact(self.config.dataset_path, type='dataset')
                self.arti_dataset.add_dir(self.config.dataset_path + '/')
                wandb.log_artifact(self.arti_dataset)
            self.arti_code = wandb.Artifact('train_script', type='code')
            self.arti_code.add_file(f'{self.config.project_name}_train.ipynb')
            wandb.log_artifact(self.arti_code)
            self.arti_results = wandb.Artifact(os.path.split(self.config.output_dir)[-1], type='results')
            self.arti_results.add_dir(self.config.output_dir)
            wandb.log_artifact(self.arti_results)
            self.wandb_finish()
        torch.cuda.empty_cache()
        print(f"🚀yl-consistency model training ends! total time:{(t2 - t1) / 3600:.3f} hours!")

    def save_loss_csv(self):
        loss_df = pd.DataFrame({'train_noise_loss' : self.logger['train_noise_loss_list']})
        if self.config.use_wandb:
            loss_table = wandb.Table(
                columns=['train_noise_loss'])
            for i in range(loss_df.shape[0]):
                loss_table.add_data(*(loss_df.iloc[i,:].tolist()))
            wandb.log({
                "loss_table": loss_table
            })
        loss_df.to_csv(os.path.join(self.config.output_dir,'loss.csv'), index=False)
    def save_loss_curve(self):
        plt.figure(figsize=(6, 6))
        plt.title('loss_curve', fontsize=15, fontweight='bold')
        # 展示网格线
        plt.grid()
        # x轴标签
        plt.xlabel('epochs', fontsize=15, fontweight='bold')
        # y轴标签
        plt.ylabel('loss', fontsize=15, fontweight='bold')
        # 绘制
        x = np.arange(0, self.config.num_epochs)
        plt.plot(x, self.logger['train_noise_loss_list'], color='blue', label='train loss')
        plt.legend(loc='upper right')  # 设置图表图例在右上角
        plt.savefig(os.path.join(self.config.output_dir, 'loss_curve.png'), bbox_inches='tight', dpi=300)
        if self.config.use_wandb:
            loss_curve = Image.open(os.path.join(self.config.output_dir, 'metrics_curves.png')).convert('RGB')
            wandb.log({
                "loss_curve": wandb.Image(loss_curve)
            })
        print('🚀show loss_curve!')
        plt.show()
    def save_model(self, epoch, mode='best'):
        checkpoint = {
            'model': self.accelerator.unwrap_model(self.model).state_dict(),
            'ema' : self.accelerator.unwrap_model(self.ema).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler' : self.lr_scheduler.state_dict(),
            'epoch': epoch + 1,
            'global_steps': self.train_global_step,
            'logger': self.logger,
        }
        if self.config.use_wandb:
            checkpoint.update({'run_id':self.run_id})
        torch.save(checkpoint, os.path.join(self.config.output_dir, f'{self.model_name}_{mode}.pt'))
        del checkpoint
    def save_config(self):
        self.config.train_dataset = None
        self.config.train_dataloader = None
        self.config.model = None
        with open(os.path.join(self.config.output_dir, f'{self.config.project_name}.pickle'), "wb") as f:
            pickle.dump(self.config, f)
    def wandb_finish(self):
        if self.config.use_wandb:
            wandb.finish()
            print('wandb has finished!')
        else:
            print('wandb did not use!')

