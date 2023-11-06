## Overview

This project focuses on the development of a binary classification model to predict the success of organizations applying for funding from the nonprofit foundation, Alphabet Soup. By utilizing machine learning techniques and neural networks, this analysis aims to assist Alphabet Soup in making more informed decisions about allocating resources to applicants.


## Table of Contents
* [Project Objective](#project-objective)
* [Data Preprocessing](#data-preprocessing)
* [Model Development](#model-development)
* [Optimization Strategies](#optimization-strategies)
* [Model Performance Analysis](#model-performance-analysis)
* [Usage](#usage)
* [Conclusion and Recommendations](#conclusion-and-recommendations)
* [Files](#files)


## Project Objective

The primary objective is to build a robust binary classification model that can predict the success or failure of organizations funded by Alphabet Soup. The goal is to enhance the organization's decision-making process, thereby increasing the success rate and impact of their philanthropic efforts.


## Data Preprocessing

### Target and Features

- **Target Variable**: The model uses "IS_SUCCESSFUL" as the target variable, which indicates the success of funding (1 for successful, 0 for not successful).

- **Feature Variables**: A range of columns from the input data, including "APPLICATION_TYPE," "AFFILIATION," "CLASSIFICATION," "USE_CASE," "ORGANIZATION," "STATUS," "INCOME_AMT," and "SPECIAL_CONSIDERATIONS," serve as feature variables.


### Data Cleanup

- The "EIN" and "NAME" columns were dropped as they were deemed non-informative for modeling.

- Unique values for each column were assessed to understand the data better.

- For columns with more than 10 unique values, data points for each unique value were examined to establish a cutoff point for binning "rare" categorical variables together into a new value, "Other."

- Categorical variables were encoded using `pd.get_dummies()`.


### Data Splitting and Scaling

- The preprocessed data was split into a feature array (X) and a target array (y).

- A StandardScaler instance was used to scale the training and testing features.



## Model Development

### Neural Network

- A neural network model was designed using TensorFlow and Keras.

- The model consists of input layers, hidden layers, and an output layer.

- Activation functions, such as "ReLU" and "sigmoid," were chosen carefully to optimize the model.


### Model Compilation and Training

- The model was compiled using an appropriate loss function and optimizer.

- It underwent training with a callback to save model weights every five epochs.


### Model Evaluation

- Model performance was evaluated using the test data to determine the loss and accuracy.


### Model Export

- The trained model was exported to an HDF5 file named "AlphabetSoupCharity.h5."


## Optimization Strategies

### Model Optimization

- The analysis involved multiple iterations with different neural network configurations, including the number of layers, neurons, activation functions, and epochs.

- Efforts were made to optimize the model by fine-tuning hyperparameters, exploring various activation functions, and considering advanced deep learning techniques.

- Despite various optimization attempts, the target model performance was not achieved in this phase.


## Model Performance Analysis

### Model Comparison

Three distinct neural network models were created, each with different configurations. Here's a brief overview:

#### Model 1
- Number of Hidden Layers: 3
- Hidden Layer 1: 9 neurons (ReLU)
- Hidden Layer 2: 5 neurons (ReLU)
- Hidden Layer 3: 2 neurons (ReLU)
- Output Layer: 1 neuron (sigmoid)
- Accuracy: ~72.64%

#### Model 2
- Number of Hidden Layers: 2
- Hidden Layer 1: 8 neurons (ReLU)
- Hidden Layer 2: 5 neurons (ReLU)
- Output Layer: 1 neuron (sigmoid)
- Accuracy: ~72.45%

#### Model 3
- Number of Hidden Layers: 4
- Hidden Layer 1: 9 neurons (ReLU)
- Hidden Layer 2: 5 neurons (tanh)
- Hidden Layer 3: 3 neurons (ReLU)
- Hidden Layer 4: 2 neurons (tanh)
- Output Layer: 1 neuron (sigmoid)
- Accuracy: ~72.80%

### Steps for Improved Performance
- Despite multiple optimization attempts, the accuracy seemed to plateau around 72-73%.

- Further strategies could include fine-tuning hyperparameters, exploring advanced deep learning techniques like dropout and batch normalization, or even considering ensemble models.

## Usage

For detailed code implementation, please refer to the Jupyter notebooks provided: "AlphabetSoupCharity_Optimization.ipynb", "AlphabetSoupCharity_Optimization2.ipynb", and "AlphabetSoupCharity_Optimization3.ipynb."

## Conclusion and Recommendations

In conclusion, the analysis aims to provide Alphabet Soup with a predictive model to enhance the allocation of funding. The current deep learning models exhibited accuracies around 72-73%, falling short of the target model performance. As an alternative approach, we recommend exploring ensemble models, such as Random Forest or Gradient Boosting, which are known to deliver superior results for classification tasks. This change may enhance the accuracy and predictive capabilities of the model when coupled with the existing data and features. Further refinement and tuning are necessary to find the most effective model for this task.

## Files

- "AlphabetSoupCharity.h5" - Trained model results.
- Jupyter notebooks: "AlphabetSoupCharity_Optimization.ipynb", "AlphabetSoupCharity_Optimization2.ipynb," and "AlphabetSoupCharity_Optimization3.ipynb."
- Additional resources: "Report.pdf" - The report detailing the analysis.
