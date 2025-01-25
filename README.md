---

# Sampling Techniques and Model Evaluation on Credit Card Fraud Detection Dataset

This project explores different sampling techniques for handling imbalanced datasets and evaluates multiple machine learning models on these samples. The dataset used is a credit card fraud detection dataset with a highly imbalanced class distribution.

## Table of Contents
- [Dataset](#dataset)
- [Sampling Techniques](#sampling-techniques)
- [Machine Learning Models](#machine-learning-models)
- [How It Works](#how-it-works)
- [Outputs](#outputs)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Rankings](#results-and-rankings)
- [Contributing](#contributing)
- [License](#license)

---

## Dataset

The dataset is publicly available and can be downloaded from the following link:
- **URL**: [Credit Card Fraud Detection Dataset](https://github.com/AnjulaMehto/Sampling_Assignment/raw/main/Creditcard_data.csv)

The target variable (`Class`) indicates whether a transaction is fraudulent (`1`) or not (`0`). The dataset is highly imbalanced, with most transactions being non-fraudulent.

---

## Sampling Techniques

The project implements the following sampling techniques to handle class imbalance:

1. **Simple Random Sampling**: Randomly selects a subset of data while maintaining class proportions.
2. **Systematic Sampling**: Selects samples at regular intervals from the dataset.
3. **Stratified Sampling**: Ensures that the sampled subset retains the class distribution of the original dataset.
4. **Cluster Sampling**: Creates clusters of data based on a feature (e.g., `Amount`) and selects a subset of clusters.
5. **Bootstrap Sampling**: Creates a sample with replacement.
6. **SMOTE (Synthetic Minority Oversampling Technique)**: Balances the dataset by generating synthetic samples for the minority class.
7. **Undersampling**: Reduces the size of the majority class to match the minority class.

---

## Machine Learning Models

The following machine learning models are evaluated:
- **Decision Tree Classifier**
- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **Gradient Boosting Classifier**

---

## How It Works

1. **Data Loading**: The dataset is loaded from the provided URL.
2. **Sampling**: Different sampling techniques are applied to balance the dataset.
3. **Model Training and Evaluation**: Each sampling method is used to train the models. Accuracy scores are calculated on a test set.
4. **Results and Rankings**: The accuracy results for all sampling methods and models are written to an output file (`output.txt`), along with rankings of models for each sampling method.

---

## Outputs

The program produces a file called `output.txt` containing:
1. Accuracy results for each sampling method and model.
2. Rankings of models based on their performance for each sampling method.

---

## Installation

### Prerequisites
- Python 3.7+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `imblearn`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sampling-techniques.git
   cd sampling-techniques
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Run the Python script:
   ```bash
   python sampling_techniques.py
   ```
2. Check the output in `output.txt`:
   ```bash
   cat output.txt
   ```

---

## Results and Rankings

The output file (`output.txt`) contains:
1. **Detailed Accuracy Results**: Accuracy of each model for all sampling methods.
2. **Model Rankings**: Models ranked by accuracy for each sampling method.

Example snippet from `output.txt`:
```
Detailed Accuracy Results:

Simple Random Sampling:
  Decision Tree: 0.9456
  Logistic Regression: 0.9723
  Random Forest: 0.9852
  SVM: 0.9708
  Gradient Boosting: 0.9867

Model Rankings by Sampling Method:

Simple Random Sampling Rankings:
  1. Gradient Boosting: 0.9867
  2. Random Forest: 0.9852
  3. Logistic Regression: 0.9723
  4. SVM: 0.9708
  5. Decision Tree: 0.9456
```

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

--- 
