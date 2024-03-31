# Required Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def preprocess_ecoli_data(df):
    # Check for missing values
    print("Missing values found. Filling missing values...")
    df_filled = df.fillna(df.median())
    print("Missing values filled successfully.")
    return df_filled

def load_dataset(file_path):
    # Read the dataset into a pandas DataFrame
    data = pd.read_csv(file_path, header=None)

    # remove rows for the minority classes
    df = data[data[7] != 'imS']
    df = data[data[7] != 'imL']

    # Fill missing values with median if any
    missing_values = data.isnull().sum()
    if missing_values.sum() != 0:
        preprocess_ecoli_data(data)

    # retrieve numpy array
    data = df.values

    # Separate features (X) and target (y)
    X = data[:, :-1]  # All columns except the last one
    y = data[:, -1]   # Only the last column

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()
    # Fit label encoder and transform target variable
    y_encoded = label_encoder.fit_transform(y)

    return X,y_encoded,label_encoder

def train_model(X, y, n_estimators=1000):
    # Machine Learning Modal to evaluate
    model = RandomForestClassifier(n_estimators=n_estimators)
    # fit the model
    model.fit(X, y)
    return model

def evaluate_Model(X,y,model,label_encoder):
    # Evaluate model on the entire dataset
    y_pred = model.predict(X)

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    print("Accuracy:", accuracy)

    # Calculate other metrics
    print("Classification Report:")
    print(classification_report(y, y_pred, target_names=label_encoder.classes_))

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

def run_Tests(label_encoder):

    print('Test Cases Result : ')

    # Define the test data
    test_data = [
        [0.67,0.39,0.48,0.50,0.36,0.38,0.46],  # expected class: cp
        [0.60,0.50,1.00,0.50,0.54,0.77,0.80],  # expected class: im
        [0.59,0.29,0.48,0.50,0.64,0.75,0.77],  # expected class: imU
        [0.65,0.51,0.48,0.50,0.66,0.54,0.33],  # expected class: om
        [0.77,0.57,1.00,0.50,0.37,0.54,0.01],  # expected class: omL
        [0.65,0.57,0.48,0.50,0.47,0.47,0.51]  # expected class: pp
    ]

    # Predict labels for the test data
    predicted_labels = []
    expected_labels = ['cp', 'im', 'imU', 'om', 'omL', 'pp']


    for i, row in enumerate(test_data):
        yhat = model.predict([row])
        label = label_encoder.inverse_transform(yhat)[0]
        predicted_labels.append(label)
        print(f'Row: {row}, Expected Class: {expected_labels[i]}, Predicted Class: {label}')

    # Calculate accuracy for the test data
    test_accuracy = accuracy_score(expected_labels, predicted_labels)
    print("Test Accuracy:", test_accuracy)


if __name__ == "__main__":
    # dataset file path
    file_path = 'ecoli.csv'

    # Load Dataset
    X,y_encoded,label_encoder = load_dataset(file_path)

    # Train Modal on data
    model = train_model(X, y_encoded)

    # Predict with trained Modal
    run_Tests(label_encoder)

    # Evaluate Modal
    evaluate_Model(X, y_encoded, model, label_encoder)
