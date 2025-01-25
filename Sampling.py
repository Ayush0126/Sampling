import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
    data["Cluster"] = data["Amount"] // (data["Amount"].max() / n_clusters)
    sampled_clusters = data["Cluster"].sample(n_clusters, random_state=42)
    sampled_data = data[data["Cluster"].isin(sampled_clusters)]
    X_sampled = sampled_data.drop(["Class", "Cluster"], axis=1)  # Drop 'Cluster' to avoid mismatch
    y_sampled = sampled_data["Class"]
    return X_sampled, y_sampled

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
# Models and Evaluation
# ----------------------------

# Define models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

# Evaluate function
def evaluate_model(model, X_train, y_train, X_test, y_test):
    if len(np.unique(y_train)) < 2:  # Handle single-class data
        return "Single class data"
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Sampling methods
sampling_methods = {
    "Simple Random": lambda: simple_random_sampling(X, y, sample_size=0.2),
    "Systematic": lambda: systematic_sampling(X, y, step=10),
    "Stratified": lambda: stratified_sampling(X, y, sample_size=0.2),
    "Clustered": lambda: cluster_sampling(data, n_clusters=5),
    "Bootstrap": lambda: bootstrap_sampling(X, y, n_samples=int(len(y) * 0.2)),
    "SMOTE": lambda: smote_oversampling(X, y),
    "Undersampling": lambda: undersampling(X, y),
}

# Store results
detailed_results = {}

# Evaluate each model on each sampling method
for method, sampler in sampling_methods.items():
    X_sampled, y_sampled = sampler()
    detailed_results[method] = {}
    for model_name, model in models.items():
        accuracy = evaluate_model(model, X_sampled, y_sampled, X_test, y_test)
        detailed_results[method][model_name] = accuracy

# Convert results to a DataFrame for easier manipulation
results_df = pd.DataFrame(detailed_results)

# ----------------------------
# Write Results to output.txt
# ----------------------------
with open("/kaggle/working/output.txt", "w") as file:
    # Write detailed results
    file.write("Detailed Accuracy Results (Model vs Sampling Methods):\n")
    file.write(results_df.to_string(index=True))
    file.write("\n\n")

    # Write rankings for each sampling method
    file.write("Model Rankings by Sampling Method:\n")
    for method in detailed_results.keys():
        file.write(f"\n{method} Rankings:\n")
        sorted_models = sorted(
            detailed_results[method].items(),
            key=lambda x: x[1] if isinstance(x[1], float) else 0,
            reverse=True,
        )
        for rank, (model_name, accuracy) in enumerate(sorted_models, start=1):
            file.write(f"  {rank}. {model_name}: {accuracy}\n")

print("Results have been written to output.txt.")
