import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

LEARNING_RATE = 0.01
BATCH_SIZE = 32
EPOCHS = 100
TEST_SIZE = 0.2
VALIDATION_SPLIT = 0.2
DROPOUT_RATE = 0.2
PATIENCE = 10
RANDOM_STATE = 42

INPUT_FILE = "/Users/sadrapga/python3/project/p0p1/p11/DataSet/TC01.in"
OUTPUT_FILE = "/Users/sadrapga/python3/project/p0p1/p11/DataSet/TC01.out"

def load_data_from_files(input_file, output_file):
    print("Loading data from files...")
    X = np.loadtxt(input_file, skiprows=1) 
    y = np.loadtxt(output_file)
    print(f"Data loaded: X shape = {X.shape}, y shape = {y.shape}")
    return X, y

def prepare_data_for_training(X, y):
    print("preparing data for training...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    # scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training set size: {len(X_train_scaled)}")
    print(f"Test set size: {len(X_test_scaled)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def create_neural_network(input_size):
    print("Creating neural network...")

    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_size,)),
        Dropout(DROPOUT_RATE),
        Dense(32, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    print("Model created successfully!")
    model.summary()
    return model

def train_model(model, X_train, y_train):
    print("Training the model...")
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=PATIENCE, 
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop],
        verbose=1
    )

    print("Training completed!")
    return history

def calculate_all_metrics(model, X_train, y_train, X_test, y_test):
    print("Calculating performance metrics...")

    y_train_pred = model.predict(X_train, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)

    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)

    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)

    return {
        'train': {'r2': train_r2, 'mse': train_mse, 'mae': train_mae, 'rmse': train_rmse},
        'test': {'r2': test_r2, 'mse': test_mse, 'mae': test_mae, 'rmse': test_rmse},
        'predictions': {'train': y_train_pred, 'test': y_test_pred}
    }

def print_model_performance(metrics):
    print("\n" + "="*50)
    print("MODEL PERFORMANCE RESULTS")
    print("="*50)

    # Training results
    train = metrics['train']
    print(f"\nTraining Set Performance:")
    print(f"  - R² Score: {train['r2']*100:.2f}%")
    print(f"  - MSE: {train['mse']:.4f}")
    print(f"  - MAE: {train['mae']:.4f}")
    print(f"  - RMSE: {train['rmse']:.4f}")

    # Test results
    test = metrics['test']
    print(f"\nTest Set Performance:")
    print(f"  - R² Score: {test['r2']*100:.2f}%")
    print(f"  - MSE: {test['mse']:.4f}")
    print(f"  - MAE: {test['mae']:.4f}")
    print(f"  - RMSE: {test['rmse']:.4f}")

    # Check for overfitting
    r2_difference = (train['r2'] - test['r2']) * 100
    print(f"\nOverfitting Check:")
    print(f"  - R² Difference (Train - Test): {r2_difference:.2f}%")

    if r2_difference > 10:
        print(" - Possible overfitting detected!")
    else:
        print(" - Good: Model generalizes well!")

def show_sample_predictions(model, X_test, y_test, num_samples=3):
    print(f"\nSample Predictions (first {num_samples} test samples):")
    print("-" * 45)

    sample_predictions = model.predict(X_test[:num_samples], verbose=0)

    for i in range(num_samples):
        predicted = sample_predictions[i][0]
        actual = y_test[i]
        error = abs(predicted - actual)

        print(f"  Sample {i+1}:")
        print(f"    Predicted: {predicted:.4f}")
        print(f"    Actual:    {actual:.4f}")
        print(f"    Error:     {error:.4f}")

def plot_training_progress(history):
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plot mae
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.title('Model MAE During Training')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(y_train, y_test, metrics):
    y_train_pred = metrics['predictions']['train']
    y_test_pred = metrics['predictions']['test']

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_train, y_train_pred, alpha=0.6, s=20, color='blue')
    min_val, max_val = y_train.min(), y_train.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Training Set (R² = {metrics["train"]["r2"]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(y_test, y_test_pred, alpha=0.6, s=20, color='orange')
    min_val, max_val = y_test.min(), y_test.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Test Set (R² = {metrics["test"]["r2"]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def run_complete_pipeline():
    print("Starting")
    print("="*50)

    try:
        X, y = load_data_from_files(INPUT_FILE, OUTPUT_FILE)

        X_train, X_test, y_train, y_test, scaler = prepare_data_for_training(X, y)

        model = create_neural_network(X_train.shape[1])

        history = train_model(model, X_train, y_train)

        metrics = calculate_all_metrics(model, X_train, y_train, X_test, y_test)

        print_model_performance(metrics)
        show_sample_predictions(model, X_test, y_test)


        print("\n Creating visualizations...")
        plot_training_progress(history)
        plot_predictions_vs_actual(y_train, y_test, metrics)


        return model, scaler, metrics

    except Exception as e:
        print(f" Error occurred: {e}")
        return None, None, None
    
if __name__ == "__main__":
    model, scaler, results = run_complete_pipeline()