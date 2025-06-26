import os
import random
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
import tensorflow as tf
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import gc

# Set global random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

def calculate_metrics(y_true, y_pred_proba):
    """
    Compute various binary classification metrics
    """
    y_pred = (y_pred_proba >= 0.5).astype(int).flatten()

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0
    precision = precision_score(y_true, y_pred) if (tp + fp) > 0 else 0
    recall = recall_score(y_true, y_pred) if (tp + fn) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    auc_score = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0
    aupr = average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0

    return {
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1 Score': f1,
        'MCC': mcc,
        'AUC': auc_score,
        'AUPR': aupr,
        'Precision': precision,
        'Recall': recall,
        'Accuracy': accuracy
    }


input_folder_path = "./Model/Data/Rice_yield.csv"       # Folder containing input .csv files
output_folder_path = "./Fine-tune"    # Folder to save the best model
os.makedirs(output_folder_path, exist_ok=True)

# Path to pretrained model
pretrained_model_path = "./pretrained/Pre-model.h5"


for file_name in os.listdir(input_folder_path):
    if not file_name.endswith(".csv"):
        continue

    data_path = os.path.join(input_folder_path, file_name)
    print(f"\nProcessing file: {file_name}")

    raw_data = pd.read_csv(data_path, header=None).values
    print(f"  Data shape: {raw_data.shape}")

    if raw_data.shape[1] != 1623:
        print(f"  Skipped: expected 1623 features, got {raw_data.shape[1]}.")
        continue
    X = raw_data
    num_samples = len(X)
    half = num_samples // 2
    y = np.array([1] * half + [0] * half)
    print(f"  Labels generated: 1s={half}, 0s={num_samples - half}")

    # Shuffle utility function
    def shuffleData(aX, ay):
        idx = list(range(len(aX)))
        random.seed(SEED)
        random.shuffle(idx)
        return aX[idx], ay[idx]

    # Shuffle training and validation data
    train_X, train_y = shuffleData(train_X, train_y)
    valid_X, valid_y = shuffleData(valid_X, valid_y)

    # Load pretrained model and freeze early layers
    base_model = load_model(pretrained_model_path)
    for layer in base_model.layers[:-5]:
        layer.trainable = False

    # Define objective function for Hyperopt
    def fine_tune_objective(params):
        lr = params["learning_rate"]
        bs = int(params["batch_size"])
        ep = int(params["epochs"])

        # Reload model and set up training
        model = load_model(pretrained_model_path)
        for layer in model.layers[:-5]:
            layer.trainable = False

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["binary_accuracy"])

        history = model.fit(
            train_X, train_y,
            validation_data=(valid_X, valid_y),
            epochs=ep,
            batch_size=bs,
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
            verbose=0
        )

        val_loss = history.history['val_loss'][-1]
        tf.keras.backend.clear_session()
        return {"loss": val_loss, "status": STATUS_OK}

    # Define hyperparameter search space
    fine_tune_space = {
        "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1e-2)),
        "batch_size": hp.choice("batch_size", [16, 32, 64]),
        "epochs": hp.choice("epochs", [10, 30, 50])
    }

    trials = Trials()
    best_params = fmin(
        fn=fine_tune_objective,
        space=fine_tune_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=np.random.RandomState(SEED)
    )
    best_params = space_eval(fine_tune_space, best_params)
    print("  Best hyperparameters:", best_params)

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    best_fold_loss = np.inf
    best_fold_model = None
    best_fold_metrics = None

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train_fold, X_valid_fold = X[train_idx], X[valid_idx]
        y_train_fold, y_valid_fold = y[train_idx], y[valid_idx]

        X_train_fold, y_train_fold = shuffleData(X_train_fold, y_train_fold)
        X_valid_fold, y_valid_fold = shuffleData(X_valid_fold, y_valid_fold)

        model = load_model(pretrained_model_path)
        for layer in model.layers[:-5]:
            layer.trainable = False

        optimizer = tf.keras.optimizers.Adam(learning_rate=best_params["learning_rate"])
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["binary_accuracy"])

        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_valid_fold, y_valid_fold),
            epochs=best_params["epochs"],
            batch_size=best_params["batch_size"],
            callbacks=[EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
            verbose=0
        )

        val_loss = history.history['val_loss'][-1]

        if val_loss < best_fold_loss:
            best_fold_loss = val_loss
            best_fold_model = model
            y_valid_proba = model.predict(X_valid_fold, batch_size=best_params["batch_size"], verbose=0)
            best_fold_metrics = calculate_metrics(y_valid_fold, y_valid_proba)

        tf.keras.backend.clear_session()
        gc.collect()

    print("  Best fold metrics based on validation loss:")
    for k, v in best_fold_metrics.items():
        print(f"    {k}: {v:.4f}")

    model_name ="Fine-model.h5"
    save_path = os.path.join(output_folder_path, model_name)
    best_fold_model.save(save_path)
    print(f"  Saved best model to: {save_path}\n")

    del best_fold_model, best_fold_metrics
    gc.collect()
    tf.keras.backend.clear_session()

print("Fine-tuning completed. One .h5 file generated per input CSV with best hyperparameters and metrics printed.")
