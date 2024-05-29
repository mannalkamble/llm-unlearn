# Selective Unlearning

This directory contains scripts, notebooks, and datasets related to the selective unlearning technique as outlined in the project on "Machine Unlearning in Large Language Models." 

## Directory Structure

- `Datasets and Evaluation prompts` - Directory containing the datasets used for training and the prompts used for evaluation.
- `dicts_new.npy`: Contains mappings from anchor terms to their generic translations.
- `evaluation.ipynb`: Jupyter notebook to evaluate the effectiveness of the unlearning process and validate the model's performance after unlearning.
- `generic_finetuning.py`: Python script for fine-tuning the model on the altered dataset where specific knowledge has been obscured or replaced.
- `prepare_dataset.py`: Python script that prepares the unlearning dataset by replacing identified specific terms with generic terms as per the provided mappings and generating alternative training labels.
- `preprocess_and_divide.ipynb`: Jupyter notebook for preprocessing the initial datasets
- `process_dataset.ipynb`: Notebook for processing datasets to align them with the requirements of selective unlearning.
- `SyntDataGen.ipynb`: Generates anchor terms to generic translations dictionaries.
- `SyntPropmtGen.ipynb`: Generates prompts to test the model's knowledge on unlearned topics, helping to assess if the unlearning has been effective.


## Usage

Follow these steps to use the tools in this directory for effective selective unlearning:

1. **Generate Replacement Dictionaries**: Use `SyntDataGen.ipynb` to create dictionaries that map specific terms to generic counterparts.

2. **Initial Fine-Tuning of the Model**: Conduct the first round of fine-tuning using `generic_finetuning.py` to train the model on the raw dataset to obtain reinforced model.

3. **Prepare the Unlearning Dataset**: Utilize `prepare_dataset.py` to generate the unlearn dataset.

4. **Second Fine-Tuning of the Model**: Perform a second round of fine-tuning with the unlearn dataset.

5. **Generate Evaluation Prompts**: Create prompts using `SyntPropmtGen.ipynb` to test the model's memory on the unlearned content.

6. **Evaluate the Unlearning Effectiveness**: Assess the effectiveness of the unlearning process with `evaluation.ipynb` by analyzing the model's responses to the prompts.

