import os
import random
import numpy as np
import pandas as pd
from keras import Input, Model
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.metrics import binary_accuracy
from keras.models import load_model
from keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, average_precision_score, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
import tensorflow as tf
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import gc

# Set global random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

def calculate_metrics(y_true, y_pred_proba):
    """
    Compute basic binary classification metrics.
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

input_folder_path = "./Model/Data/Rice_yield.csv"
output_folder_path = "./Fusion"
os.makedirs(output_folder_path, exist_ok=True)

model_path1 = "./Fine-model1.h5"
model_path2 = "./Fine-model2.h5"

base1 = load_model(model_path1)
layer_name1 = base1.layers[-4].name
intermediate_model1 = Model(inputs=base1.input, outputs=base1.get_layer(layer_name1).output)

base2 = load_model(model_path2)
layer_name2 = base2.layers[-4].name
intermediate_model2 = Model(inputs=base2.input, outputs=base2.get_layer(layer_name2).output)

def fusion_model():
    """
    Define fusion model that takes concatenated features of size 4096.
    """
    inputs = Input(shape=(4096,))
    x = Dense(2048, activation='relu', kernel_regularizer=l2(0.001))(inputs)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(x)
    output = Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=output)

def shuffle_data(X, y):
    """
    Shuffle X and y using the global random seed.
    """
    idx = list(range(len(X)))
    random.seed(RANDOM_SEED)
    random.shuffle(idx)
    return X[idx], y[idx]

for file_name in os.listdir(input_folder_path):
    if not file_name.endswith(".csv"):
        continue

    data_path = os.path.join(input_folder_path, file_name)
    print(f"\nProcessing {file_name}")

    raw = pd.read_csv(data_path, header=None).values
    X_raw = raw
    y_raw = np.array([1] * (len(raw) // 2) + [0] * (len(raw) // 2))

    feat1 = intermediate_model1.predict(X_raw, verbose=0)
    feat2 = intermediate_model2.predict(X_raw, verbose=0)
    X = np.concatenate((feat1, feat2), axis=1)
    y = y_raw

    print(f"  Concatenated feature shape: {X.shape}")

    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
    train_X, train_y = shuffle_data(train_X, train_y)
    valid_X, valid_y = shuffle_data(valid_X, valid_y)

    def fine_tune_objective(params):
        lr = params['learning_rate']
        batch_size = int(params['batch_size'])
        epochs = int(params['epochs'])

        model = fusion_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[binary_accuracy])

        history = model.fit(
            train_X, train_y,
            validation_data=(valid_X, valid_y),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=0
        )

        val_loss = history.history['val_loss'][-1]
        tf.keras.backend.clear_session()
        return {'loss': val_loss, 'status': STATUS_OK}

    # Hyperparameter search space
    fine_tune_space = {
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
        'batch_size': hp.choice('batch_size', [16, 32, 64]),
        'epochs': hp.choice('epochs', [10, 30, 50])
    }

    trials = Trials()
    best = fmin(
        fn=fine_tune_objective,
        space=fine_tune_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=np.random.RandomState(RANDOM_SEED)
    )
    best_params = space_eval(fine_tune_space, best)
    print(f"  Best hyperparameters: {best_params}")

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    best_fold_loss = np.inf
    best_fold_model = None
    best_fold_metrics = None

    for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(X)):
        X_train_fold, X_valid_fold = X[train_idx], X[valid_idx]
        y_train_fold, y_valid_fold = y[train_idx], y[valid_idx]

        X_train_fold, y_train_fold = shuffle_data(X_train_fold, y_train_fold)
        X_valid_fold, y_valid_fold = shuffle_data(X_valid_fold, y_valid_fold)

        model = fusion_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[binary_accuracy])

        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_valid_fold, y_valid_fold),
            epochs=best_params['epochs'],
            batch_size=best_params['batch_size'],
            callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
            verbose=0
        )

        val_loss = history.history['val_loss'][-1]
        if val_loss < best_fold_loss:
            best_fold_loss = val_loss
            best_fold_model = model
            y_valid_proba = model.predict(X_valid_fold, batch_size=best_params['batch_size'], verbose=0)
            best_fold_metrics = calculate_metrics(y_valid_fold, y_valid_proba)

        tf.keras.backend.clear_session()
        gc.collect()

    print("  Best fold metrics (by validation loss):")
    for metric, value in best_fold_metrics.items():
        print(f"    {metric}: {value:.4f}")

    model_name ="Fusion-model.h5"
    save_path = os.path.join(output_folder_path, model_name)
    best_fold_model.save(save_path)
    print(f"  Saved best model: {save_path}")

    # Cleanup
    del best_fold_model, best_fold_metrics
    gc.collect()
    tf.keras.backend.clear_session()

print("Fusion process completed. One .h5 file generated per input CSV; best hyperparameters and metrics printed.")
