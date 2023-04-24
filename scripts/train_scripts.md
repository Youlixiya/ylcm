# train_scripts
1.unconditional_cm_cifar10_32

```shell
python train.py --image_size 32 --batch_size 256 --num_samples 64 --sample_steps 5 --num_epochs 100 --project_name unconditional_cm_cifar10_32 --precision 32 --exp cm/unconditional_cm_cifar10_32 --model CIFAR10UNet2DModel --dataset_name CIFAR10CMDataset --use_wandb
```