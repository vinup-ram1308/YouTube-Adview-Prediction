# YouTube Adview Prediction

This project aims to predict the number of ad views on YouTube videos based on various features using machine learning models. The project involves pre-processing the data, training the model, and making predictions on the test data.

## Dataset

The dataset consists of YouTube video features and the corresponding ad views. The dataset is divided into training and test sets.

- **Training Set**: Contains both features and the target variable (ad views).
- **Test Set**: Contains only the features, and the task is to predict the ad views.

## Project Objectives

The main objectives of this project are:
- To preprocess the training and test datasets.
- To train a machine learning model to predict ad views.
- To make predictions on the test dataset and save the results to a CSV file.

## Implementation Steps

### Data Preprocessing

1. **Load the Data**: Load the training and test datasets.

2. **Convert Duration to Seconds**: Convert the duration feature from ISO 8601 format to seconds.

3. **Handle Missing Values**: Check for and handle any missing values in the datasets.

4. **Feature Engineering**: Create new features or modify existing ones to improve the model's performance.

5. **Scaling Features**: Scale the features to ensure they are on a similar scale.(Using MinMaxScaler)

### Model Training

1. **Train-Test Split**: Split the training data into training and validation sets.

2. **Train the Model**: Train a machine learning model (e.g., Linear Regression, Random Forest) on the training data.

3. **Evaluate the Model**: Evaluate the model's performance on the validation set using appropriate metrics (e.g., RMSE).

4. **Save the Model**: Save the trained model using the `joblib` library.

### Making Predictions

1. **Load the Model**: Load the saved model using the `joblib` library.

2. **Preprocess the Test Data**: Apply the same preprocessing steps to the test data as done for the training data.

3. **Make Predictions**: Use the model to make predictions on the test data.

4. **Save Predictions**: Save the predictions to a CSV file named `PredictedAdview.csv`.

## Dependencies

- numpy
- pandas
- scikit-learn
- joblib

## Installation

1. Clone the repository to your local machine:

```bash
git clone https://github.com/vinup-ram1308/YouTube-Adview-Prediction

```
2. Navigate to the project directory:

```bash
cd Youtube-Adview-Prediction
```
3. Install the required dependencies.


## Usage

1. Train the Model:

Run the Jupyter Notebook to preprocess the data, train the model, and save the model
```bash
jupyter notebook ML_Project.ipynb
```

2. Make predictions

Run the Python script to load the model, preprocess the test data, and save the predictions:
```bash
python predict.py
```
## Results

The predicted ad views for the test dataset will be saved in a CSV file named `PredictedAdview.csv`.
