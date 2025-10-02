import gc
import os
import random
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, average_precision_score, f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
import tensorflow as tf
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def calculate_metrics(y_true, y_pred, y_pred_proba):
    TP = np.sum((y_pred >= 0.5) & (y_true == 1))
    TN = np.sum((y_pred < 0.5) & (y_true == 0))
    FP = np.sum((y_pred >= 0.5) & (y_true == 0))
    FN = np.sum((y_pred < 0.5) & (y_true == 1))

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0

    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    aupr = auc(recall, precision)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision_score_val = TP / (TP + FP) if (TP + FP) > 0 else 0

    performance_metrics = {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'F1 Score': f1,
        'MCC': mcc,
        'AUPR': aupr,
        'AUC': auc_score,
        'Precision': precision_score_val,
        'Accuracy': accuracy
    }
    return performance_metrics


input_folder_path = "./Model/Data/Rice_yield.csv"
output_folder_path = "./Fine-tune"
os.makedirs(output_folder_path, exist_ok=True)

pretrained_model_path = "./Pre-model.h5"

best_hyperparameters = []

file_name = os.path.basename(input_folder_path)
input_file_path = input_folder_path

new_input_data = pd.read_csv(input_file_path, header=None)
new_X = new_input_data.values
new_y = np.array([1] * (int(len(new_X)) // 2) + [0] * (int(len(new_X)) // 2))

kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_X, valid_X, train_y, valid_y = train_test_split(new_X, new_y, test_size=0.2, random_state=42)


def shuffleData(X, y):
    index = [i for i in range(len(X))]
    random.seed(2510)
    random.shuffle(index)
    X = X[index]
    y = y[index]
    return X, y


train_X, train_y = shuffleData(train_X, train_y)
valid_X, valid_y = shuffleData(valid_X, valid_y)


def fine_tune_objective(params):
    learning_rate = params['learning_rate']
    batch_size = int(params['batch_size'])
    epochs = params['epochs']

    model = load_model(pretrained_model_path)
    for layer in model.layers[:-5]:
        layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    history = model.fit(
        train_X, train_y,
        validation_data=(valid_X, valid_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=0, restore_best_weights=True)],
        verbose=0
    )

    val_loss = min(history.history['val_loss'])
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return {'loss': val_loss, 'status': STATUS_OK}


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
    rstate=np.random.RandomState(42)
)

best_params = space_eval(fine_tune_space, best)
print("Best hyperparameters:", best_params)

best_learning_rate = best_params['learning_rate']
best_batch_size = best_params['batch_size']
best_epochs = best_params['epochs']

fold_results = []
for fold_idx, (train_index, valid_index) in enumerate(kf.split(new_X)):
    print(f"Processing Fold {fold_idx + 1}/{kf.n_splits}")

    train_X_fold, valid_X_fold = new_X[train_index], new_X[valid_index]
    train_y_fold, valid_y_fold = new_y[train_index], new_y[valid_index]

    train_X_fold, train_y_fold = shuffleData(train_X_fold, train_y_fold)
    valid_X_fold, valid_y_fold = shuffleData(valid_X_fold, valid_y_fold)

    model = load_model(pretrained_model_path)
    for layer in model.layers[:-5]:
        layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=best_learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    model.fit(
        train_X_fold, train_y_fold,
        validation_data=(valid_X_fold, valid_y_fold),
        epochs=best_epochs,
        batch_size=best_batch_size,
        callbacks=[early_stopping],
        verbose=1)

    predictions_proba = model.predict(valid_X_fold)
    predictions_bin = (predictions_proba > 0.5).astype(int).flatten()

    metrics = calculate_metrics(
        y_true=valid_y_fold,
        y_pred=predictions_bin,
        y_pred_proba=predictions_proba.flatten()
    )

    results_after = {
        'Fold': fold_idx + 1,
        'TP': metrics['TP'],
        'TN': metrics['TN'],
        'FP': metrics['FP'],
        'FN': metrics['FN'],
        'Accuracy': round(metrics['Accuracy'], 4),
        'Precision': round(metrics['Precision'], 4),
        'Sensitivity': round(metrics['Sensitivity'], 4),
        'F1 Score': round(metrics['F1 Score'], 4),
        'AUC': round(metrics['AUC'], 4),
        'AUPR': round(metrics['AUPR'], 4),
        'MCC': round(metrics['MCC'], 4),
        'Specificity': round(metrics['Specificity'], 4)
    }

    fold_results.append(results_after)

    prediction_df = pd.DataFrame({
        'True Label': valid_y_fold,
        'Predicted Probability': predictions_proba.flatten(),
        'Predicted Label': predictions_bin
    })
    prediction_file_name = os.path.splitext(file_name)[0] + f'_fold_{fold_idx + 1}_predictions.csv'
    prediction_df.to_csv(os.path.join(output_folder_path, prediction_file_name), index=False)

    del model, train_X_fold, valid_X_fold, train_y_fold, valid_y_fold
    tf.keras.backend.clear_session()
    gc.collect()

df = pd.DataFrame(fold_results)
df.loc['Mean'] = df.mean(numeric_only=True)
df.loc['Std'] = df.std(numeric_only=True)

output_file_name = os.path.splitext(file_name)[0] + '_kfold_results.xlsx'
df.to_excel(os.path.join(output_folder_path, output_file_name), index=True, engine='openpyxl')

print("Training final model on the full dataset with best hyperparameters...")

final_model = load_model(pretrained_model_path)
for layer in final_model.layers[:-5]:
    layer.trainable = False

final_optimizer = tf.keras.optimizers.Adam(learning_rate=best_learning_rate)
final_model.compile(optimizer=final_optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

final_model_path = os.path.join(output_folder_path, f"{os.path.splitext(file_name)[0]}_fine-tune_model.h5")

final_model.fit(
    new_X, new_y,
    epochs=best_epochs,
    batch_size=best_batch_size,
    callbacks=[EarlyStopping(monitor='loss', patience=10, verbose=1, restore_best_weights=True)],
    verbose=1
)

final_model.save(final_model_path)
print(f"Final model saved at {final_model_path}")

del final_model, df
tf.keras.backend.clear_session()
gc.collect()

print("Fine-tuning completed. Only final models have been saved.")
