# Scripts for Evaluation

This repository folder contains various scripts and utilities used to evaluate the **SAM3** and **SAM3D** models across multiple benchmarks and datasets.

## Directory Structure

- **`eval/`**: Python scripts and configurations for evaluating on different dataset tiers (e.g., SACO Gold, Silver, VEval), including utilities for data downloading and preprocessing.
- **`server/`**: Bash scripts useful for server environments to set up tools, download datasets (ODinW, Roboflow 100), obtain model checkpoints, and execute large-scale evaluation runs.
- **`table1/`**: Convenience shell scripts to reproduce primary baseline results for COCO, LVIS, SACO datasets, etc.
- **Root Scripts**: Root level scripts contain utilities for extracting evaluation results (to CSV format), measuring inference speeds, qualitative testing, and running missing job states.

## Evaluation Outputs

You can download the collected evaluation results/outputs here:

- 📦 [output.zip](https://drive.google.com/drive/folders/1a0p5mU3oL5s-qaDVqfA6iu0DDXoGefrr?usp=sharing)
