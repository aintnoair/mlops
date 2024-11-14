# train.py

import argparse
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from data_module import GLUEDataModule
from model import GLUETransformer
import wandb

if not wandb.api.api_key:
    wandb.login()

parser = argparse.ArgumentParser(description="Train model on GLUE task")
parser.add_argument("--checkpoint_dir", type=str, default="models", help="Directory to save checkpoints")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Model name or path")
parser.add_argument("--task_name", type=str, default="mrpc", help="GLUE task name")
parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
parser.add_argument("--train_batch_size", type=int, default=32, help="Training batch size")
parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
parser.add_argument("--wandb_group", type=str, default="", help="Group this run is part of")
args = parser.parse_args()

# Seed everything for reproducibility
seed_everything(42)

wandb_logger = WandbLogger(
    project="mlops_p2_containers",
    group=args.wandb_group if args.wandb_group else None
)

# Initialize the data module
dm = GLUEDataModule(
    model_name_or_path=args.model_name,
    task_name=args.task_name,
    max_seq_length=args.max_seq_length,
    train_batch_size=args.train_batch_size,
    eval_batch_size=args.eval_batch_size,
)

# Set up the data
dm.setup("fit")

# Initialize the model
model = GLUETransformer(
    model_name_or_path=args.model_name,
    num_labels=dm.num_labels,
    eval_splits=dm.eval_splits,
    task_name=args.task_name,
    learning_rate=args.lr,
    weight_decay=args.weight_decay,
    warmup_steps=args.warmup_steps,
    train_batch_size=args.train_batch_size,
    eval_batch_size=args.eval_batch_size,
)

# Set up the trainer
trainer = Trainer(
    max_epochs=args.epochs,
    accelerator="auto",  # Automatically selects CPU or GPU if available
    devices=1,  # Use one device (adjust based on your resources)
    logger=wandb_logger,  # Log metrics to W&B
    default_root_dir=args.checkpoint_dir,  # Directory to save checkpoints
)

# Run training
trainer.fit(model, datamodule=dm)

# Finalize W&B logging
wandb.finish()
