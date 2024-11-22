# P2 Containers: Containerized Machine Learning Training Project

This project demonstrates how to containerize a machine learning training script using Docker. 
The goal is to ensure a consistent environment across development and production by running the 
project inside a Docker container. The training script uses PyTorch Lightning and WandB for logging 
and supports flexible configuration of hyperparameters.

`Disclaimer`: I developed and tested this project on an M1 Macbook Pro, results may vary when using different 
 a different OS or x86 processor architecture.

## Project Overview

This project containerizes a training pipeline for a language model, built for consistency and portability. 
The key files and their purposes are:
- **train.py**: Main script to run the training, configurable through command-line arguments.
- **data_module.py** and **model.py**: Modules defining the data loading and model architecture.
- **Dockerfile**: Docker configuration to build an image with all dependencies.
- **requirements.txt**: Lists all Python dependencies needed for the project.

## Prerequisites

To run this project, you’ll need:
- **Docker** installed on your machine.
- A **WandB** account (for logging) if you intend to log experiment results.

## Step 1: Clone the Repository

Start by cloning this repository to your local machine:

```bash
git clone https://github.com/aintnoair/mlops.git
cd p2_containers
```

## Step 2: Build the Docker Image
Next, build the Docker image using the Dockerfile provided:
```bash
docker build -t p2_containers .
```

This command creates a Docker image named `p2_containers`. The build process installs all dependencies 
listed in `requirements.txt` and sets up the environment for training.

## Step 3: Run the Docker Container
Once the image is built, you can start a training run inside the container. Use the following command, adjusting 
the hyperparameters as needed:

```bash
docker run -it p2_containers --checkpoint_dir models --lr 3.500853360234228e-5 --warmup_steps 85 --weight_decay 0.28973691202547286
```

### Command-Line Arguments
The following command-line arguments are available:

| Argument            | Type   | Default                     | Description                        |
|---------------------|--------|-----------------------------|------------------------------------|
| `--checkpoint_dir`  | `str`  | `"models"`                  | Directory to save checkpoints      |
| `--lr`              | `float`| `1e-3`                      | Learning rate                      |
| `--model_name`      | `str`  | `"distilbert-base-uncased"` | Model name or path              |
| `--task_name`       | `str`  | `"mrpc"`                    | GLUE task name                     |
| `--max_seq_length`  | `int`  | `128`                       | Maximum sequence length            |
| `--train_batch_size`| `int`  | `32`                        | Training batch size                |
| `--eval_batch_size` | `int`  | `32`                        | Evaluation batch size              |
| `--epochs`          | `int`  | `3`                         | Number of epochs                   |
| `--weight_decay`    | `float`| `0.0`                       | Weight decay                       |
| `--warmup_steps`    | `int`  | `0`                         | Number of warmup steps             |
| `--wandb_group`     | `str`  | `None`                      | Group this run is part of          |


### Example with WandB Group
You can specify a WandB group name for the run as follows:

```bash
docker run -it p2_containers --wandb_group experiment_group
```

### WandB API Key Configuration
If WandB prompts you for an API key, you can enter it interactively. 
1. Choose option 2 to use an existing WandB Account for logging
2. Paste your API key into the terminal and press the enter key to start and log the run. 

### Additional Tips
**Rebuild the Docker Image**: If you make changes to the code (e.g., `train.py`), you’ll need to
rebuild the Docker image for changes to take effect:

```bash
docker build -t p2_containers .
```

**Cleaning Up:** After testing, you can remove stopped containers and unused images to free up disk space:
```bash
docker system prune
```

### Repository Structure
```bash
p2_containers/
├── dockerfile            # Docker configuration file for building the image
├── README.md             # This guide
├── train.py              # Main script to run training
├── data_module.py        # Data module for data loading and preparation
├── model.py              # Model definition for training
└── requirements.txt      # Python dependencies
```

### Troubleshooting
#### Docker Image Size
If the Docker image is large, try using smaller base images (e.g., `python:3.11-alpine`) or combine RUN statements in 
the Dockerfile to reduce the image layers.

#### Slow Training
Docker may be slower than running directly on your local machine. If you’re using a GPU, ensure that Docker is 
configured for GPU support. Alternatively, allocate more resources (CPUs and memory) to Docker using the following run
command:
```bash
docker run --gpus all -it p2_containers --checkpoint_dir models --lr 3.500853360234228e-5 --warmup_steps 85 --weight_decay 0.28973691202547286
```

For any additional questions please hesitate to reach out :)

