import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_recall_fscore_support, roc_curve, precision_recall_curve)
import streamlit as st

# Loading the Data 
def load_data():
    features = pd.read_csv('C:/Users/kanipriya/Desktop/Predictive model/UCI HAR Dataset/UCI HAR Dataset/features.txt', sep='\s+', header=None)[1].tolist()
    features = pd.Series(features)
    features = features + '_' + features.groupby(features).cumcount().astype(str)
    
    X_train = pd.read_csv('C:/Users/kanipriya/Desktop/Predictive model/UCI HAR Dataset/UCI HAR Dataset/train/X_train.txt', sep='\s+', header=None, names=features)
    y_train = pd.read_csv('C:/Users/kanipriya/Desktop/Predictive model/UCI HAR Dataset/UCI HAR Dataset/train/y_train.txt', sep='\s+', header=None, names=['Activity'])

    X_test = pd.read_csv('C:/Users/kanipriya/Desktop/Predictive model/UCI HAR Dataset/UCI HAR Dataset/test/X_test.txt', sep='\s+', header=None, names=features)
    y_test = pd.read_csv('C:/Users/kanipriya/Desktop/Predictive model/UCI HAR Dataset/UCI HAR Dataset/test/y_test.txt', sep='\s+', header=None, names=['Activity'])

    return X_train, X_test, y_train, y_test


# Preprocessing Data
def preprocess_data(X_train, X_test):
    non_zero_var_columns = X_train.var() != 0
    X_train = X_train.loc[:, non_zero_var_columns]
    X_test = X_test.loc[:, non_zero_var_columns]
    return X_train, X_test

# Training the Model
def train_model(X_train, y_train, model_type='RandomForest', n_estimators=100, max_depth=None, C=1.0, kernel='linear', n_neighbors=5, max_iter=1000):
    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, max_depth=max_depth)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    elif model_type == 'SVM':
        model = SVC(C=C, kernel=kernel, probability=True, random_state=42)
    elif model_type == 'KNN':
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    elif model_type == 'DecisionTree':  
        model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    model.fit(X_train, y_train.values.ravel())
    return model

# Evaluateing the Model
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Classification Report
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    st.write(f"**Accuracy for {model_name}:** {acc}")
    st.write(f"**Precision:** {precision}")
    st.write(f"**Recall:** {recall}")
    st.write(f"**F1 Score:** {f1}")
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    st.pyplot(plt.gcf())
    return y_pred, acc

# ROC and Precision-Recall Curves
def plot_roc_pr_curves(model, X_test, y_test):
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_proba, pos_label=1)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    st.pyplot(plt.gcf())

    precision, recall, _ = precision_recall_curve(y_test, y_proba, pos_label=1)
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    st.pyplot(plt.gcf())

# Feature Importance for Random Forest
def plot_feature_importance(model, features):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.barh(range(len(indices)), importances[indices], color="b", align="center")
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel("Relative Importance")
        st.pyplot(plt.gcf())

# Model Comparison
def compare_models(X_train, y_train, X_test, y_test):
    models = ['RandomForest', 'LogisticRegression', 'SVM', 'KNN', 'DecisionTree']
    accuracies = {}
    
    for model_type in models:
        model = train_model(X_train, y_train, model_type=model_type)
        _, acc = evaluate_model(model, X_test, y_test, model_type)
        accuracies[model_type] = acc

    plt.figure(figsize=(10, 6))
    plt.bar(accuracies.keys(), accuracies.values(), color='b')
    plt.title('Model Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    st.pyplot(plt.gcf())

# Streamlit App
def streamlit_app():
    st.title("Human Activity Recognition (HAR) Classification")

    X_train, X_test, y_train, y_test = load_data()
    X_train, X_test = preprocess_data(X_train, X_test)

    # Model Selection
    model_choice = st.selectbox('Select Model', ['RandomForest', 'LogisticRegression', 'SVM', 'KNN', 'DecisionTree'])
    st.write(f"You selected {model_choice}")

    # Initializing all hyperparameters to None
    n_estimators = None
    max_depth = None
    C = None
    kernel = None
    n_neighbors = None
    max_iter = None

    # Hyperparameter tuning based on the selected model
    if model_choice == 'RandomForest':
        n_estimators = st.slider('Number of Estimators', min_value=10, max_value=200, value=100, step=10)
        max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=None)
    elif model_choice == 'DecisionTree':
        max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=None)
    elif model_choice == 'LogisticRegression':
        C = st.slider('Regularization Strength (C)', min_value=0.01, max_value=10.0, value=1.0)
        max_iter = st.slider('Max Iterations', min_value=100, max_value=2000, value=1000)
    elif model_choice == 'SVM':
        C = st.slider('Regularization Strength (C)', min_value=0.01, max_value=10.0, value=1.0)
        kernel = st.selectbox('Kernel', ['linear', 'rbf', 'poly'])
    elif model_choice == 'KNN':
        n_neighbors = st.slider('Number of Neighbors', min_value=1, max_value=20, value=5)

    # Train Selected Model
    if st.button("Train Model"):
        model = train_model(X_train, y_train, model_choice, n_estimators, max_depth, C, kernel, n_neighbors, max_iter)
        y_pred, acc = evaluate_model(model, X_test, y_test, model_choice)
        st.write(f"Accuracy of {model_choice}: {acc}")

        if model_choice == 'RandomForest':
            st.write("Feature Importances for RandomForest:")
            plot_feature_importance(model, X_train.columns)

        plot_roc_pr_curves(model, X_test, y_test)

    # Button for Comparing All Models
    if st.button("Compare All Models"):
        st.write("Comparing Models:")
        compare_models(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    streamlit_app()
