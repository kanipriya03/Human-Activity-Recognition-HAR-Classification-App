# Human Activity Recognition (HAR) Classification

This project involves building a machine learning model to classify human activities using data from the UCI Human Activity Recognition (HAR) dataset. Various classification algorithms such as Random Forest, Logistic Regression, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Decision Trees are implemented and compared.


## Prerequisites

The following libraries are required:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `streamlit`

## Project Structure

- **Data Preprocessing**: Load and clean the dataset, removing columns with zero variance.
- **Model Training**: Train models including Random Forest, Logistic Regression, SVM, KNN, and Decision Trees.
- **Model Evaluation**: Evaluate the performance using metrics such as accuracy, precision, recall, F1 score, confusion matrix, ROC curve, and precision-recall curve.
- **Feature Importance**: For Random Forest, feature importance is plotted.
- **Model Comparison**: Compare the accuracy of all the models.

## Key Functions

1. **`load_data()`**: Loads the training and testing data from the dataset.
2. **`preprocess_data(X_train, X_test)`**: Removes columns with zero variance.
3. **`train_model()`**: Trains the selected model with user-defined hyperparameters.
4. **`evaluate_model()`**: Evaluates the model using accuracy, precision, recall, F1 score, and plots confusion matrix.
5. **`plot_roc_pr_curves()`**: Plots ROC and precision-recall curves.
6. **`plot_feature_importance()`**: Displays feature importance for Random Forest models.
7. **`compare_models()`**: Compares the accuracy of all the models.

## Streamlit App

The project is built as a web app using Streamlit, where users can:
- Select a classification model from the list.
- Tune hyperparameters based on the selected model.
- Train the model and view evaluation metrics and curves.
- Compare the performance of all models.

## How to Run
Installation Steps
1.Clone or Download the Repository:
  `git clone <repository_url>
   cd <repository_folder>`
2. Install the necessary libraries using `pip`:
   `pip install pandas numpy matplotlib seaborn scikit-learn streamlit`
3. Download the UCI HAR dataset and place it in the appropriate directory.
4. Run the Streamlit application:
    `streamlit run <your_script_name>.py`
5.The app will open in a browser where you can select models, tune hyperparameters, train models, and visualize results.

# How to Use

1. **Model Selection and Hyperparameter Tuning**:
   - Select a machine learning model from the dropdown list (Random Forest, Logistic Regression, SVM, KNN, or Decision Tree).
   - Customize hyperparameters using the provided sliders. For example:
     - Random Forest: Choose the number of estimators and the maximum depth of the trees.
     - Logistic Regression: Adjust the regularization strength (C) and the number of iterations.
     - SVM: Select the kernel type and adjust the regularization strength.
     - KNN: Choose the number of neighbors.
   - Each model has its own specific hyperparameters, adjustable based on user needs.

2. **Training the Model**:
   - Once hyperparameters are selected, click the **Train Model** button.
   - The app will train the chosen model and display evaluation metrics, including:
     - Accuracy
     - Precision
     - Recall
     - F1 Score
   - For models like Random Forest, feature importance is also plotted.

3. **Evaluating Performance**:
   - After training, confusion matrices, ROC curves, and precision-recall curves are automatically displayed.
   - These plots give you a better understanding of the model’s predictive performance.

4. **Comparing All Models**:
   - Click the **Compare All Models** button to train all available models on the dataset.
   - A bar chart of model accuracies will be displayed to compare performance across models.

## Available Models

- **Random Forest Classifier**: A powerful ensemble method using decision trees.
- **Logistic Regression**: A linear model for binary and multiclass classification.
- **Support Vector Machine (SVM)**: A robust algorithm for linear and non-linear classification problems.
- **K-Nearest Neighbors (KNN)**: A simple, instance-based learning method.
- **Decision Tree Classifier**: A non-parametric model based on decision trees.

## Model Evaluation

After training the model, the app evaluates the model based on the following metrics:

- **Accuracy**: The overall percentage of correctly predicted activities.
- **Precision**: How many of the predicted positives were actually positive.
- **Recall**: How many actual positives were identified correctly.
- **F1 Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A visual representation of true versus predicted labels.
- **ROC Curve**: A plot of the true positive rate against the false positive rate.
- **Precision-Recall Curve**: Shows the trade-off between precision and recall.

### Feature Importance (Random Forest):

If the Random Forest model is selected, the app displays the most important features that contribute to the classification process.

## Model Comparison

Click the **Compare All Models** button to:

- Train all models (Random Forest, Logistic Regression, SVM, KNN, Decision Tree).
- Display a comparison of their accuracy on a bar chart.

## Dataset Information

The UCI HAR dataset consists of accelerometer and gyroscope sensor readings collected from smartphones worn by participants performing various activities. The dataset includes:

- **Train Set**: Used to train the model.
- **Test Set**: Used to evaluate model performance.
- **Features**: Sensor data representing various physical activities like walking, sitting, standing, etc.

Ensure that the dataset is downloaded and organized correctly for the app to function properly.

