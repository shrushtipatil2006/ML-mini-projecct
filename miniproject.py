
# ============================================================
# CREDIT CARD FRAUD DETECTION USING MACHINE LEARNING
# Algorithms:
# 1. Multiple Linear Regression
# 2. Logistic Regression
# 3. K-Nearest Neighbors (KNN)
# 4. Support Vector Machine (SVM)
# 5. AdaBoost
# 6. Gradient Boosting
# ============================================================

# STEP 1: INSTALL REQUIRED LIBRARIES
!pip install -q pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

# STEP 2: IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

print("Libraries imported successfully!")

df = pd.read_csv("small_credit_card_fraud_dataset.csv")

print("\nFIRST FIVE ROWS:")
print(df.head())

print("\nDATASET INFORMATION:")
df.info()

print("\nDATASET SHAPE:", df.shape)

print("\nMISSING VALUES:", df.isnull().sum().sum())

print("\nDUPLICATE ROWS:", df.duplicated().sum())

print("\nCLASS DISTRIBUTION:")
print(df["Class"].value_counts())

# STEP 4: DATA PREPROCESSING

# Remove duplicate records
df = df.drop_duplicates().copy()

# Remove rows with missing target values
df = df.dropna(subset=["Class"])

# Separate input features and target
X = df.drop("Class", axis=1)
y = df["Class"].astype(int)

print("\nFEATURE SHAPE:", X.shape)
print("TARGET SHAPE:", y.shape)

print("\nCLASS PERCENTAGE:")
print(y.value_counts(normalize=True) * 100)

# STEP 5: VISUALIZE CLASS DISTRIBUTION
plt.figure(figsize=(7, 5))
sns.countplot(x=y)

plt.title("Genuine vs Fraudulent Transactions")
plt.xlabel("Class (0 = Genuine, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.show()

# STEP 6: TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING DATA:", X_train.shape)
print("TESTING DATA:", X_test.shape)

# STEP 7: LIMIT GENUINE TRAINING SAMPLES
# Retain all fraud samples and sample up to 20,000 genuine records.
# This reduces training time.

genuine_indices = y_train[y_train == 0].index
fraud_indices = y_train[y_train == 1].index

max_genuine = min(20000, len(genuine_indices))

genuine_sample = np.random.RandomState(42).choice(
    genuine_indices,
    size=max_genuine,
    replace=False
)

selected_indices = np.concatenate([
    genuine_sample,
    fraud_indices
])

X_train_selected = X_train.loc[selected_indices]
y_train_selected = y_train.loc[selected_indices]

print("\nBEFORE SMOTE:")
print(y_train_selected.value_counts())

# STEP 8: DEFINE MACHINE LEARNING MODELS

models = {

    "Multiple Linear Regression":
        LinearRegression(),

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ),

    "SVM":
        LinearSVC(
            max_iter=5000,
            random_state=42
        ),

    "AdaBoost":
        AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.5,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
}

print("\nALL SIX MODELS CREATED SUCCESSFULLY!")

# STEP 9: TRAIN AND EVALUATE ALL MODELS

results = []
trained_models = {}
predictions = {}

for name, model in models.items():

    print("\n" + "=" * 55)
    print("TRAINING:", name)
    print("=" * 55)

    # Create pipeline
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(
            random_state=42,
            k_neighbors=5
        )),
        ("model", model)
    ])

    # Train model
    pipeline.fit(
        X_train_selected,
        y_train_selected
    )

    trained_models[name] = pipeline

    # Predict test data
    raw_predictions = pipeline.predict(X_test)

    # Convert regression predictions into binary classes
    if name == "Multiple Linear Regression":
        y_pred = (raw_predictions >= 0.5).astype(int)
    else:
        y_pred = raw_predictions.astype(int)

    predictions[name] = y_pred

    # Obtain prediction scores
    if hasattr(pipeline, "predict_proba"):
        y_scores = pipeline.predict_proba(X_test)[:, 1]

    elif hasattr(pipeline, "decision_function"):
        y_scores = pipeline.decision_function(X_test)

    else:
        y_scores = raw_predictions

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test, y_scores
    )

    pr_auc = average_precision_score(
        y_test, y_scores
    )

    # Store results
    results.append({
        "Algorithm": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    })

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))
    print("PR-AUC   :", round(pr_auc, 4))


# STEP 10: DISPLAY FINAL COMPARISON TABLE

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1-Score",
    ascending=False
)

print("\nFINAL ALGORITHM COMPARISON:")
print(results_df.to_string(index=False))

results_df.to_csv(
    "fraud_detection_comparison.csv",
    index=False
)

# STEP 11: PERFORMANCE COMPARISON GRAPH

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC",
    "PR-AUC"
]

results_df.set_index("Algorithm")[metrics].plot(
    kind="bar",
    figsize=(14, 7)
)

plt.title("Performance Comparison of ML Algorithms")
plt.xlabel("Algorithms")
plt.ylabel("Performance Score")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1.05)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# STEP 12: CONFUSION MATRICES

fig, axes = plt.subplots(
    2, 3,
    figsize=(16, 10)
)

axes = axes.flatten()

for i, (name, y_pred) in enumerate(predictions.items()):

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Genuine", "Fraud"],
        yticklabels=["Genuine", "Fraud"],
        ax=axes[i]
    )

    axes[i].set_title(name)
    axes[i].set_xlabel("Predicted Class")
    axes[i].set_ylabel("Actual Class")

plt.tight_layout()
plt.show()

# STEP 13: FALSE POSITIVE AND FALSE NEGATIVE COMPARISON

error_results = []

for name, y_pred in predictions.items():

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    error_results.append({
        "Algorithm": name,
        "True Positive": tp,
        "True Negative": tn,
        "False Positive": fp,
        "False Negative": fn
    })

error_df = pd.DataFrame(error_results)

print("\nERROR COMPARISON:")
print(error_df.to_string(index=False))

# STEP 14: ADABOOST VS OTHER ALGORITHMS

comparison = results_df[
    ["Algorithm", "Precision", "Recall", "F1-Score", "PR-AUC"]
]

print("\nADABOOST VS OTHER ALGORITHMS:")
print(comparison.to_string(index=False))

comparison.set_index("Algorithm").plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("AdaBoost vs Other ML Algorithms")
plt.xlabel("Algorithms")
plt.ylabel("Performance Score")
plt.xticks(rotation=30, ha="right")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# STEP 15: IDENTIFY BEST MODEL

best_model_name = results_df.iloc[0]["Algorithm"]

print("\n" + "=" * 55)
print("BEST MODEL BASED ON F1-SCORE")
print("=" * 55)

print("Best Algorithm:", best_model_name)

print("\nBEST MODEL PERFORMANCE:")
print(results_df.iloc[0].to_string())

# Classification report
print("\nCLASSIFICATION REPORT:")

print(
    classification_report(
        y_test,
        predictions[best_model_name],
        target_names=["Genuine", "Fraud"],
        zero_division=0
    )
)

# STEP 16: SAVE FINAL RESULTS

results_df.to_csv(
    "final_model_comparison.csv",
    index=False
)

error_df.to_csv(
    "fraud_detection_errors.csv",
    index=False
)

print("\nPROJECT EXECUTED SUCCESSFULLY!")
print("Comparison results saved.")
print("Error comparison saved.")
