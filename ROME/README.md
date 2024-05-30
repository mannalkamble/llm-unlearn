# Rank-One Model Editing (ROME)

This collection includes scripts, Jupyter notebooks, and datasets used to explore the Rome technique for large language models, as described in the project on "Machine Unlearning."

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Directory Structure](#directory-structure)

## Overview

ROME (Rank-One Model Editing) is a framework designed to enable modifications and updates to large language models' knowledge bases without the need for extensive retraining. This project provides tools and methodologies to perform efficient model editing, ensuring that models can adapt to new information or remove outdated knowledge.

## Installation

To install the necessary dependencies, please use the following:
 
```
git clone https://github.com/mannalkamble/llm-unlearn.git
cd llm-unlearn/ROME
pip install -r requirements.txt
```

## Usage
To run the experiments and utilize the tools provided in this repository, follow the steps below:
1. **Configuration**: Update the globals.yml file with your desired settings and hyperparameters.
2. **Running Scripts**: Use the provided Jupyter notebook run.ipynb for interactive experimentation or run the scripts directly from the command line.
   
## Directory Structure 

Here is an overview of the directory structure of this repository:
```
llm-unlearn/ROME/
├── baselines/         # Baseline models and comparisons
├── dsets/             # Datasets for training and evaluation
├── experiments/       # Experimental results and configurations
├── hparams/           # Hyperparameter configurations
├── notebooks/         # Jupyter notebooks for experiments and analysis
├── rome/              # Core ROME framework code
├── scripts/           # Scripts for running experiments and tools
├── util/              # Utility functions and helpers
├── CITATION.cff       # Citation information
├── LICENSE            # License information
├── README.md          # Project README file
├── globals.yml        # Global configuration file
└── run.ipynb          # Main Jupyter notebook for running experiments
```
