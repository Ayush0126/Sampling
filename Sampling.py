import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Load the dataset
url = "https://github.com/AnjulaMehto/Sampling_Assignment/raw/main/Creditcard_data.csv"
data = pd.read_csv(url)

# Separate features and target
X = data.drop("Class", axis=1)
y = data["Class"]

# ----------------------------
# Sampling Techniques
# ----------------------------

# 1. Simple Random Sampling
def simple_random_sampling(X, y, sample_size):
    # Combine features and target into one DataFrame
    data = pd.concat([X, y], axis=1)
    # Stratify sampling by class
    data_sampled = data.groupby("Class", group_keys=False).apply(
        lambda x: x.sample(frac=sample_size, random_state=42)
    )
    return data_sampled.drop("Class", axis=1), data_sampled["Class"]

# 2. Systematic Sampling
def systematic_sampling(X, y, step):
    # Combine features and target into one DataFrame
    data = pd.concat([X, y], axis=1)
    sampled_indices = []
    for class_label in data["Class"].unique():
        class_data = data[data["Class"] == class_label]
        sampled_indices.extend(class_data.iloc[::step].index)
    data_sampled = data.loc[sampled_indices]
    return data_sampled.drop("Class", axis=1), data_sampled["Class"]


# 3. Stratified Sampling
def stratified_sampling(X, y, sample_size):
    X_train, X_sample, y_train, y_sample = train_test_split(
        X, y, stratify=y, test_size=sample_size, random_state=42
    )
    return X_sample, y_sample

# 4. Cluster Sampling
def cluster_sampling(data, n_clusters=5):
    # Create artificial clusters based on features
    data["Cluster"] = data["Amount"] // (data["Amount"].max() / n_clusters)
    sampled_clusters = data["Cluster"].sample(n_clusters, random_state=42)
    sampled_data = data[data["Cluster"].isin(sampled_clusters)]
    return sampled_data.drop("Cluster", axis=1)

# 5. Bootstrap Sampling
def bootstrap_sampling(X, y, n_samples=None):
    n_samples = n_samples or len(y)
    X_sample, y_sample = resample(X, y, replace=True, n_samples=n_samples, random_state=42)
    return X_sample, y_sample

# 6. Oversampling (SMOTE)
def smote_oversampling(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# 7. Undersampling
def undersampling(X, y):
    data = pd.concat([X, y], axis=1)
    majority_class = data[data["Class"] == 0]
    minority_class = data[data["Class"] == 1]
    majority_downsampled = resample(
        majority_class, replace=False, n_samples=len(minority_class), random_state=42
    )
    downsampled_data = pd.concat([majority_downsampled, minority_class])
    return downsampled_data.drop("Class", axis=1), downsampled_data["Class"]

# ----------------------------
# Experiment Setup
# ----------------------------

# Sample size and step size for sampling
sample_size = 0.2
step_size = 10  # for systematic sampling

# Generate sampled datasets
X_simple, y_simple = simple_random_sampling(X, y, sample_size)
X_systematic, y_systematic = systematic_sampling(X, y, step_size)
X_stratified, y_stratified = stratified_sampling(X, y, sample_size)
clustered_data = cluster_sampling(data, n_clusters=10)  # Increase clusters if needed
X_clustered = clustered_data.drop("Class", axis=1)
y_clustered = clustered_data["Class"]

X_bootstrap, y_bootstrap = bootstrap_sampling(X, y, n_samples=int(len(y) * sample_size))
X_smote, y_smote = smote_oversampling(X, y)
X_undersampled, y_undersampled = undersampling(X, y)

# ----------------------------
# Models and Evaluation
# ----------------------------

# General function for model evaluation
# Updated evaluation function with class validation
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Check if y_train contains at least two classes
    if len(np.unique(y_train)) < 2:
        return "Single class data"
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Train-test split for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Store accuracy and rankings
detailed_results = {}

# Evaluate models on all sampling methods
for method, (X_sampled, y_sampled) in {
    "Simple Random": (X_simple, y_simple),
    "Systematic": (X_systematic, y_systematic),
    "Stratified": (X_stratified, y_stratified),
    "Clustered": (X_clustered, y_clustered),
    "Bootstrap": (X_bootstrap, y_bootstrap),
    "SMOTE": (X_smote, y_smote),
    "Undersampling": (X_undersampled, y_undersampled),
}.items():
    detailed_results[method] = {}
    for model_name, model in models.items():
        # Evaluate the model and handle single-class cases
        accuracy = evaluate_model(model, X_sampled, y_sampled, X_test, y_test)
        detailed_results[method][model_name] = accuracy

# Calculate rankings for each sampling method
rankings = {}
for method, model_results in detailed_results.items():
    valid_results = {model: acc for model, acc in model_results.items() if isinstance(acc, (float, int))}
    ranked_models = sorted(valid_results.items(), key=lambda x: x[1], reverse=True)
    rankings[method] = [f"{rank + 1}. {model}: {accuracy:.4f}" for rank, (model, accuracy) in enumerate(ranked_models)]

# Write detailed results and rankings to output.txt
output_file = "/kaggle/working/output.txt"
with open(output_file, "w") as file:
    # Write accuracy results
    file.write("Detailed Accuracy Results:\n")
    for method, model_results in detailed_results.items():
        file.write(f"\n{method} Sampling:\n")
        for model_name, accuracy in model_results.items():
            if isinstance(accuracy, str):
                file.write(f"  {model_name}: {accuracy}\n")
            else:
                file.write(f"  {model_name}: {accuracy:.4f}\n")

    # Write model rankings
    file.write("\nModel Rankings by Sampling Method:\n")
    for method, ranked_list in rankings.items():
        file.write(f"\n{method} Sampling Rankings:\n")
        for rank in ranked_list:
            file.write(f"  {rank}\n")

print(f"Results and rankings have been written to {output_file}")
