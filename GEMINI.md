# GEMINI.md

## Project Overview

This project is a sign language translation pipeline that uses a combination of a Graph Convolutional Network (GCN), a Transformer, and a Large Language Model (LLM). The goal is to translate a stream of chunked 2D keypoints from COCO WholeBody into text.

The pipeline is structured as follows:

1.  **Dataset Pipeline:** Raw pose data is chunked, augmented, and normalized.
2.  **Multi-Part GCN Stage:** A Uni-GCN model processes the keypoints for different body parts.
3.  **Chunk Transformer Encoder:** A transformer encoder processes the GCN output.
4.  **LLM Wrapper:** A large language model generates the final text output.

The project uses `deepspeed` for distributed training and is built on top of `pytorch`. The training process features a sophisticated two-stage strategy, including a visual pre-training stage that uses a powerful InfoNCE contrastive loss with in-batch negatives and projection heads to align visual and text representations, preventing representation collapse.

## Building and Running

### Dependencies

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

### Training

To start training, run the `start_train.sh` script. You can optionally provide paths to the embedding, training, and deepspeed configuration files.

```bash
bash start_train.sh [EMB_CFG] [TRAIN_CFG] [DS_CFG]
```

-   `EMB_CFG`: Path to the embedding configuration file. Defaults to `configs/embedding_default.yaml`.
-   `TRAIN_CFG`: Path to the training configuration file. Defaults to `configs/train_default.yaml`.
-   `DS_CFG`: Path to the DeepSpeed configuration file. Defaults to `configs/ds_config.json`.

The number of GPUs can be set with the `NGPU` environment variable.

### Inference

To run inference, use the `start_infer.sh` script. You need to provide the path to the checkpoint directory.

```bash
bash start_infer.sh CHECKPOINT_DIR [CONFIG] [DS_CONFIG] [MODE] [NPY] [OUTPUT]
```

-   `CHECKPOINT_DIR`: Path to the checkpoint directory.
-   `CONFIG`: Path to the training configuration file. Defaults to `configs/train_default.yaml`.
-   `DS_CONFIG`: Path to the DeepSpeed configuration file. Defaults to `configs/ds_config_bf16.json`.
-   `MODE`: Inference mode. Defaults to `test`.
-   `NPY`: Path to a numpy file for single-file inference.
-   `OUTPUT`: Path to the output file.

## Development Conventions

*   The project uses `deepspeed` for training. Make sure it is installed and configured correctly.
*   Configuration is managed through YAML files in the `configs` directory.
*   The `pipeline.md` file should be kept up-to-date with any changes to the pipeline.
*   Follow Linus-style patches: small, direct changes with clear intent.
