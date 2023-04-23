import argparse
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_size', type=int, default=32,
                        help='resolution of the image')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='training batch_size')
    parser.add_argument('--save_image_epochs', type=int, default=1,
                        help='save image epochs')
    parser.add_argument('--save_model_epochs', type=int, default=1,
                        help='save model epochs')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='sample numbers')
    parser.add_argument('--sample_steps', type=int, default=5,
                        help='sample steps in the inference')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='max number of training epochs')
    parser.add_argument('--warm_epochs', type=int, default=3,
                        help='number of warm up epochs')
    parser.add_argument('--nc', type=int, default=10,
                        help='number of dataset class')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--optimizer', type=str, default='RAdam',
                        help='optimizer type')
    parser.add_argument('--data_std', type=float, default=0.5,
                        help='data std')
    parser.add_argument('--eps', type=float, default=0.002,
                        help='min time step')
    parser.add_argument('--T', type=float, default=80.,
                        help='max time step')
    parser.add_argument('--s0', type=int, default=2,
                        help='s0')
    parser.add_argument('--s1', type=int, default=150,
                        help='s1')
    parser.add_argument('--rho', type=int, default=7,
                        help='rho')
    parser.add_argument('--use_ema', action="store_true",
                        help='use ema')
    parser.add_argument('--mu0', type=float, default=0.9,
                        help='mu0')
    parser.add_argument('--precision', type=str, default='32',
                        help='precision')

    parser.add_argument('--exp', type=str, default='cm/unconditional_cm_cifar10_32',
                        help='experiment path')
    parser.add_argument('--seed', type=int, default=0,
                        help='set everything same seed')
    parser.add_argument('--loss_fn', type=str, default="LearnedPerceptualImagePatchSimilarity(net_type='vgg')",
                        choices=["LearnedPerceptualImagePatchSimilarity(net_type='vgg')",
                                 'MSELoss()',
                                 'L1Loss()'],
                        help='loss function')
    parser.add_argument('--model', type=str, default='CIFAR10UNet2DModel',
                        choices=['CIFAR10UNet2DModel', 'MyUNet2DModel'],
                        help='choose UNetModel')
    parser.add_argument('--resume_ckpt_path', type=str, default=None,
                        help='ckpt path use to resume training environment')

    parser.add_argument('--project_name', type=str, default='unconditional_cm_cifar10_32',
                        help='wandb project name, only be used when use_wandb is True')
    parser.add_argument('--conditional', action="store_true",
                        help='whether use conditional generation or not')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='dataset root path')
    parser.add_argument('--dataset_name', type=str, default='CIFAR10CMDataset',
                        choices=['CMDataset', 'CIFAR10CMDataset'],
                        help='dataset name')
    parser.add_argument('--modes', type=list, default=['train', 'valid', 'test'],
                        choices=[['train'], ['valid'], ['test'],
                                 ['train', 'valid'], ['train', 'test'],
                                 ['valid', 'test'], ['train', 'valid', 'test']],
                        help='dataset name')
    parser.add_argument('--use_wandb', action="store_true",
                        help='whether use wandb or not')
    parser.add_argument('--wandb_id', type=str, default=None,
                        help='wandb id which can be used to resume training environment')
    parser.add_argument('--push_to_hub', action="store_true",
                        help='whether to push model to huggingface hub')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of workers of dataloader')
    parser.add_argument('--max_nums', type=int, default=None,
                        help='max number of dataset')

    return parser.parse_args()

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
        optimizer=config.optimizer,
        data_std=config.data_std,
        eps=config.eps,
        T=config.T,
        s0=config.s0,
        s1=config.s1,
        rou=config.rou,
        use_ema=config.use_ema,
        mu0=config.mu0,
        precision=config.precision,  # `no` for float32, `fp16` for automatic mixed precision
        output_dir=config.output_dir,  # the model namy locally and on the HF Hub
        seed=config.seed,
        loss_fn=config.loss_fn,
        model=config.model,
        project_name=config.project_name,
        conditional=config.conditional,
        dataset_name=config.dataset_name,
        workers=config.workers,
    )
