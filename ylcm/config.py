import json
import os
from dataclasses import dataclass
import numpy as np
import torch
from diffusers import UNet2DModel
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose
from ylcm.dataset import CIFAR10CMDataset, CMDataset
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders
@dataclass(repr=True)
class CMConfig:
    image_size: int  # the generated image resolution
    train_batch_size: int
    save_image_epochs : int
    save_model_epochs : int
    num_samples : int
    sample_steps : int
    num_epochs: int
    nc: int
    warm_epochs: int
    learning_rate: float
    momentum: float  # SGD momentum/Adam beta1
    weight_decay: float
    optimizer: str
    data_std: float
    eps: float
    T: float
    s0: int
    s1: int
    rou: int
    use_ema : bool
    mu0 : float # initial_ema_decay
    mixed_precision: str  # `no` for float32, `fp16` for automatic mixed precision
    output_dir: str  # the model namy locally and on the HF Hub
    seed: int
    loss_fn : str
    model_args: dict
    model: Module
    resume_ckpt_path: str
    project_name: str
    conditional : bool
    index2label_file_path: str
    index2label: dict
    train_txt: str
    valid_txt: str
    test_txt: str
    dataset_path: str
    dataset_name: str
    train_images_file_list: list
    train_labels_list: list
    valid_images_file_list: list
    valid_labels_list: list
    test_images_file_list: list
    test_labels_list: list
    train_dataset: Dataset
    train_dataloader: DataLoader
    use_wandb: bool
    push_to_hub : bool
    workers: int
    transforms: Compose
    max_nums: int
unconditional_cifar10_cmconfig_dict = dict(
    image_size = 32,  # the generated image resolution
    train_batch_size = 256,
    save_image_epochs = 1,
    save_model_epochs =1,
    num_samples = 16,
    sample_steps = 10,
    num_epochs = 50,
    nc = 10,
    warm_epochs = 3,
    learning_rate = 1e-4,
    momentum = 0,  # SGD momentum/Adam beta1
    weight_decay =  0,
    optimizer = 'RAdam',
    data_std = 0.5,
    eps = 0.002,
    T = 80,
    s0 = 2,
    s1 = 150,
    rou = 7,
    use_ema = True,
    mu0 = 0.9,
    mixed_precision = 'fp16',  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'cm' , # the model namy locally and on the HF Hub
    seed = 213,
    loss_fn = "LearnedPerceptualImagePatchSimilarity(net_type='vgg')",
    # loss_fn = 'L1Loss()',
    model_args=dict(
        model_type='UNet2DModel',
        model_config=dict(
            sample_size=32,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 256, 256, 256),  # the number of output channes for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D"
            ),
        )),
    model = None,
    resume_ckpt_path = None,
    project_name = 'unconditional_cm_cifar10_32',
    conditional = False,
    index2label_file_path = None,
    index2label = None,
    train_txt = None,
    valid_txt = None,
    test_txt = None,
    dataset_path = None,
    dataset_name = 'CIFAR10CMDataset',
    train_images_file_list = None,
    train_labels_list = None,
    valid_images_file_list = None,
    valid_labels_list = None,
    test_images_file_list = None,
    test_labels_list = None,
    train_dataset = None,
    train_dataloader = None,
    use_wandb = False,
    push_to_hub = False,
    workers = 0,
    transforms = None,
    max_nums = None
)
unconditional_oxfordflowers102_cmconfig_dict = dict(
    image_size = 128,  # the generated image resolution
    train_batch_size = 32,
    save_image_epochs = 1,
    save_model_epochs =1,
    num_samples = 16,
    sample_steps = 10,
    num_epochs = 30,
    nc = 102,
    warm_epochs = 3,
    learning_rate = 1e-4,
    momentum = 0,  # SGD momentum/Adam beta1
    weight_decay =  0,
    optimizer = 'RAdam',
    data_std = 0.5,
    eps = 0.002,
    T = 80,
    s0 = 2,
    s1 = 150,
    rou = 7,
    use_ema = True,
    mu0 = 0.9,
    mixed_precision = 'fp16',  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = 'cm' , # the model namy locally and on the HF Hub
    seed = 0,
    loss_fn = "LearnedPerceptualImagePatchSimilarity(net_type='vgg')",
    # loss_fn = 'L1Loss()',
    model_args=dict(
        model_type='UNet2DModel',
        model_config=dict(
            sample_size=128,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D"
            ),
        )),
    model = None,
    resume_ckpt_path = None,
    project_name = 'unconditional_cm_oxfordflowers102_128',
    conditional = True,
    index2label_file_path = 'oxford-102-flowers/index2label.json',
    index2label = None,
    train_txt = 'train.txt',
    valid_txt = 'valid.txt',
    test_txt = 'test.txt',
    dataset_path = 'oxford-102-flowers',
    dataset_name = 'CMDataset',
    train_images_file_list = None,
    train_labels_list = None,
    valid_images_file_list = None,
    valid_labels_list = None,
    test_images_file_list = None,
    test_labels_list = None,
    train_dataset = None,
    train_dataloader = None,
    use_wandb = False,
    push_to_hub = False,
    workers = 0,
    transforms = None,
    max_nums = None
)
def get_save_path(config):
    output_dir = config.output_dir
    project_name = config.project_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    dir_list = os.listdir(output_dir)
    if dir_list:
        exp_list = [int(dir.split('_')[-1]) for dir in dir_list if project_name in dir]
        if exp_list:
            cur = max(exp_list) + 1
        else:
            cur = 0
    else:
        cur = 0
    save_dir_final = os.path.join(output_dir, f'{project_name}_{cur}')
    config.output_dir = save_dir_final
def get_model(config):
    config.model = eval(config.model_args['model_type'])(**config.model_args['model_config'])
def get_data_list(data_path, data_list_file_path):
    labels, images = [], []
    file_path = os.path.join(data_path, data_list_file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            images.append(os.path.join(data_path, line[0]))
            labels.append(int(line[1]))
    return images, labels
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = True  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构
def get_dataloader(config):
    same_seeds(config.seed) #设置随机种子使实验结果可复现
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + config.seed)
    train_dataset = eval(config.dataset_name)(config,transforms = config.transforms, max_nums = config.max_nums)
    train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True, pin_memory=PIN_MEMORY, num_workers=config.workers, generator=generator)
    config.train_dataset = train_dataset
    config.train_dataloader = train_dataloader
def get_image_path(config):
    if config.dataset_path:
        config.train_images_file_list, config.train_labels_list = get_data_list(config.dataset_path,config.train_txt)
        config.valid_images_file_list, config.valid_labels_list = get_data_list(config.dataset_path, config.valid_txt)
        config.test_images_file_list, config.test_labels_list = get_data_list(config.dataset_path, config.test_txt)
        config.images_file_list = config.train_images_file_list+config.valid_images_file_list+config.test_images_file_list
        config.labels_list = config.train_labels_list+config.valid_labels_list+config.test_labels_list
    else:
        cif = CIFAR10('.', False, download=True)
        config.index2label = {i:cls for i, cls in enumerate(cif.classes)}
def get_wandb_dict(config):
    config.wandb_dict=dict(
        image_size=config.image_size,  # the generated image resolution
        train_batch_size=config.train_batch_size,
        save_image_epochs=config.save_image_epochs,
        save_model_epochs=config.save_model_epochs,
        num_samples=config.num_samples,
        sample_steps=config.sample_steps,
        num_epochs=config.num_epochs,
        nc=config.nc,
        warm_epochs=config.warm_epochs,
        learning_rate=config.learning_rate,
        momentum=config.momentum,  # SGD momentum/Adam beta1
        weight_decay=config.weight_decay,
        optimizer=config.optimizer,
        data_std=config.data_std,
        eps=config.eps,
        T=config.T,
        s0=config.s0,
        s1=config.s1,
        rou=config.rou,
        use_ema=config.use_ema,
        mu0=config.mu0,
        mixed_precision=config.mixed_precision,  # `no` for float32, `fp16` for automatic mixed precision
        output_dir=config.output_dir,  # the model namy locally and on the HF Hub
        seed=config.seed,
        loss_fn=config.loss_fn,
        model_args=config.model_args,
        project_name=config.project_name,
        conditional=config.conditional,
        dataset_name=config.dataset_name,
        workers=config.workers,
    )
def get_index2label(config):
    with open(config.index2label_file_path, "r", encoding="utf-8") as f:
        index2index = json.load(f)
        label2index = {value: int(key) for key, value in index2index.items()}
        index2label = {value: key for key, value in label2index.items()}
    config.index2label = index2label
def get_config(config_dict, config_type):
    config = config_type(**config_dict)
    get_save_path(config)
    get_model(config)
    if config.dataset_name in ['CMDataset'] and config.conditional:
        get_index2label(config)
    get_image_path(config)
    get_wandb_dict(config)
    return config