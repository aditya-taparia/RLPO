# RLPO-Reinforcement Learning based Preference Optimization

We present a novel method for generating concept-based explanations autonomously, without the need for predefined human inputs or retrieval systems. Our approach employs an optimized reinforcement learning (RL) algorithm that continuously refines its search and optimizes the surrounding states at each step. This technique is capable of generating concepts beyond the input domain, resulting in more comprehensive and innovative explanations.

## Structure

This project is organized into several key directories:

- `gen_xai_pipeline.py`: Contains code and models related to the classification tasks.
- `config.py`: Contains code and models for image generation tasks.
- `inference.py`: Contains code and models for summarization tasks.
- `Dataset/`: Dataset creation and other plotting functions.
- `Extra/`: Datasets used in different tasks.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Install with pip

```bash
pip install -r requirments.txt
```

### Getting dataset

After installing the necessary requirements, you need to perform the following steps to prepare your dataset:

1. Navigate to the `datasets` folder and open the `dataset.ipynb` file. This Jupyter notebook contains instructions for downloading the classification dataset from Kaggle.
2. To download the dataset, you must have a Kaggle API key. Follow [this link](https://www.kaggle.com/docs/api) for instructions on how to create and generate your API key.
3. After downloading the zip file, extract it into the `Dataset` folder. 
4. Then, execute the remaining code cells in the `dataset.ipynb` file as instructed. These steps will create additional necessary files within the `Dataset` folder itself.

## Configuration Parameters for Model Explanation

In `config.py`, you can set various training parameters to explain any class for any model. Below are the detailed explanations of each parameter:

## Parameters

#### `CLASS_NAME`
- **Description:** Index of the class in ImageNet.
- **Usage:** This parameter specifies which class you want to get an explanation for. It is a string value of the class number.
- **Example:** `CLASS_NAME = "340"`

#### `REMOVE_WORD`
- **Description:** Set this to any word you want removed during preprocessing.
- **Usage:** Use this to exclude specific words from the preprocessing phase.
- **Example:** `REMOVE_WORD = "zebra"`

#### `QA_MODEL`
- **Description:** VQA model for keyword extraction.
- **Usage:** This model is used for extracting keywords relevant to the explanation.
- **Example:** `QA_MODEL = "Salesforce/blip-vqa-capfilt-large"`

#### `QUESTIONS`
- **Description:** Questions used for keyword extraction.
- **Usage:** A list of questions to assist in extracting relevant keywords from the VQA model.
- **Example:** `QUESTIONS = ["What is the object?", "Describe the scene."]`

#### `LAYERS`
- **Description:** The layer you want to explain.
- **Usage:** Specify the layer in the model that you want to generate explanations for.
- **Example:** `LAYERS = ['inception4e']`

#### `CLASS_TO_BE_EXPLAINED`
- **Description:** Index of the class you want to explain.
- **Usage:** This parameter identifies the specific class for which the explanation is generated.
- **Example:** `CLASS_TO_BE_EXPLAINED = 340`

#### `MODEL_TO_EXPLAIN`
- **Description:** Define the model from which you are taking the layer.
- **Usage:** Specify the model that contains the layer to be explained.
- **Example:** `MODEL_TO_EXPLAIN = "torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1)"`

#### `GEN_MODEL_NAME`
- **Description:** Generation model for exploration.
- **Usage:** The model used for generating explanations during exploration. It can use any Hugging Face model.
- **Example:** `GEN_MODEL_NAME = "runwayml/stable-diffusion-v1-5"`

#### `TOTAL_RL_STEP`
- **Description:** Timestep for DQN algorithm.
- **Usage:** Set the number of timesteps for the reinforcement learning algorithm.
- **Example:** `TOTAL_RL_STEP = 1000`

### Example Configuration

```python
CLASS_NAME = "340"
REMOVE_WORD = "zebra"
QA_MODEL = "Salesforce/blip-vqa-capfilt-large"
QUESTIONS = ["What is the object?", "Describe the scene."]
LAYERS = ['inception4e']
CLASS_TO_BE_EXPLAINED = 340
MODEL_TO_EXPLAIN = "torchvision.models.googlenet(weights=torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1)"
GEN_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
TOTAL_RL_STEP = 1000
