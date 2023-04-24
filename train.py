import os
from ylcm.config import get_args
from ylcm.consistency import Consistency
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
def main():
    config = get_args()
    seed_everything(config.seed)
    if config.use_wandb:
        if config.wandb_id:
            wandb_logger = WandbLogger(
                    project=config.project_name,
                    log_model=True,
                    id=config.wandb_id,
                    resume="must")
        else:
             wandb_logger = WandbLogger(
                    project=config.project_name,
                    log_model=True)
    else:
        wandb_logger=None

    if config.resume_ckpt_path:
        consistency = Consistency.load_from_checkpoint(
            checkpoint_path=config.resume_ckpt_path
        )
    else:
        consistency = Consistency(config)

    trainer = Trainer(
            accelerator="auto",
            logger=wandb_logger,
            callbacks=[
                ModelCheckpoint(
                    dirpath=os.path.join(config.exp,"ckpt"),
                    filename="epoch:d",
                    save_top_k=3,
                    monitor="loss",
                )
            ],
            max_epochs=config.num_epochs,
            precision=config.precision,
            log_every_n_steps=1,
            benchmark=True,
        )
    trainer.fit(consistency, ckpt_path=config.resume_ckpt_path)
if __name__ == '__main__':
    main()