# Machine Unlearning in Large Language Models

This repository contains the code for the project on "Machine Unlearning in Large Language Models" 

## Overview

The project synthesizes methods from the papers "Who’s Harry Potter? Approximate Unlearning in LLMs" and "Locating and Editing Factual Associations in GPT" to develop a framework capable of implementing two distinct unlearning approaches:
1. **Selective Unlearning** - Employs reinforced model predictions to selectively remove knowledge of specific content.
2. **Rank-One Model Editing (ROME)** - Utilizes direct manipulation of model weights to update factual associations precisely.

## Repository Structure

- `/Selective_Unlearning`: Contains scripts and notebooks implementing the selective unlearning process.
- `/ROME`: Includes the implementation of Rank-One Model Editing (ROME) for precise factual modifications in models.

## Authors

- **Mannal Kamble** - mk8475@nyu.edu
- **Karthvik Sarvade** - ks6807@nyu.edu


## Acknowledgments

- Thanks to Professor Gustavo Sandoval for his guidance and mentorship throughout this project.
- Inspired by the methodologies detailed in recent unlearning research papers.
