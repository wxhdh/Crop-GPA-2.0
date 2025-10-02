import os
import random
import numpy as np
import pandas as pd
from keras.layers import Conv1D, Dense, Dropout, Flatten, Input, Concatenate, Lambda, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import gc

RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Detected GPUs: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Memory growth enabled for GPUs.")
    except RuntimeError as e:
        print(f"GPU setup failed: {e}")
else:
    print("No GPUs detected.")


def shuffleData(X, y, seed=RANDOM_SEED):
    index = np.arange(len(X))
    np.random.seed(seed)
    np.random.shuffle(index)
    return X[index], y[index]


def CropGPA2_0(filters_onehot, kernel_size, filters_vec_2, filters_shape, dropout_rate, seed=RANDOM_SEED):
    tf.random.set_seed(seed)
    Features = Input(shape=(1626, 1))

    onehot = Lambda(lambda x: x[:, :164, :])(Features)
    vec_2 = Lambda(lambda x: x[:, 164:328, :])(Features)
    bert = Lambda(lambda x: x[:, 328:1096, :])(Features)
    shape = Lambda(lambda x: x[:, 1096:, :])(Features)

    onehot = Conv1D(filters=filters_onehot[0], kernel_size=kernel_size, strides=1, padding="same",
                    kernel_regularizer=l2(0.001))(onehot)
    onehot = Conv1D(filters=filters_onehot[1], kernel_size=kernel_size, strides=1, padding="same",
                    kernel_regularizer=l2(0.001))(onehot)
    onehot = MaxPooling1D(pool_size=3, strides=1, padding='valid')(onehot)
    onehot = Conv1D(filters=filters_onehot[2], kernel_size=kernel_size, strides=1, padding="same",
                    kernel_regularizer=l2(0.001))(onehot)
    onehot = Flatten()(onehot)

    vec_2 = Conv1D(filters=filters_vec_2[0], kernel_size=kernel_size, strides=1, padding="same",
                   kernel_regularizer=l2(0.001))(vec_2)
    vec_2 = Conv1D(filters=filters_vec_2[1], kernel_size=kernel_size, strides=1, padding="same",
                   kernel_regularizer=l2(0.001))(vec_2)
    vec_2 = MaxPooling1D(pool_size=3, strides=1, padding='valid')(vec_2)
    vec_2 = Conv1D(filters=filters_vec_2[2], kernel_size=kernel_size, strides=1, padding="same",
                   kernel_regularizer=l2(0.001))(vec_2)
    vec_2 = Flatten()(vec_2)

    bert = Flatten()(bert)
    bert = Dense(768, activation='relu', kernel_regularizer=l2(0.001))(bert)
    bert = Dropout(dropout_rate)(bert)
    bert = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(bert)
    bert = Dropout(dropout_rate)(bert)
    bert = Flatten()(bert)

    shape = Conv1D(filters=filters_shape[0], kernel_size=kernel_size, strides=1, padding="same",
                   kernel_regularizer=l2(0.001))(shape)
    shape = Conv1D(filters=filters_shape[1], kernel_size=kernel_size, strides=1, padding="same",
                   kernel_regularizer=l2(0.001))(shape)
    shape = MaxPooling1D(pool_size=4, strides=1, padding='valid')(shape)
    shape = Conv1D(filters=filters_shape[2], kernel_size=kernel_size, strides=1, padding="same",
                   kernel_regularizer=l2(0.001))(shape)
    shape = Flatten()(shape)

    concatenated = Concatenate(axis=-1)([onehot, vec_2, bert, shape])
    flattened_output = Flatten()(concatenated)

    dense1Layer = Dense(2048, activation='relu', name='dense1', kernel_regularizer=l2(0.001))(flattened_output)
    dense2Layer = Dense(256, activation='relu', name='dense2', kernel_regularizer=l2(0.001))(dense1Layer)
    dense3layer = Dense(32, activation='relu', name='dense3', kernel_regularizer=l2(0.001))(dense2Layer)

    pred = Dense(1, activation='sigmoid', name='dense4')(dense3layer)

    model = Model(inputs=Features, outputs=pred)
    return model


space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1e-2)),
    'dropout_rate': hp.uniform('dropout_rate', 0.1, 0.5),
    'batch_size': hp.choice('batch_size', [64, 128, 256]),
    'epochs': hp.choice('epochs', [20, 50, 100]),
    'filters_onehot_1': hp.choice('filters_onehot_1', [2, 4, 8]),
    'filters_onehot_2': hp.choice('filters_onehot_2', [4, 8, 16]),
    'filters_onehot_3': hp.choice('filters_onehot_3', [8, 16, 32]),
    'filters_vec_2_1': hp.choice('filters_vec_2_1', [2, 4, 8]),
    'filters_vec_2_2': hp.choice('filters_vec_2_2', [4, 8, 16]),
    'filters_vec_2_3': hp.choice('filters_vec_2_3', [8, 16, 32]),
    'filters_shape_1': hp.choice('filters_shape_1', [2, 4, 8]),
    'filters_shape_2': hp.choice('filters_shape_2', [4, 8, 16]),
    'filters_shape_3': hp.choice('filters_shape_3', [8, 16, 32]),
    'kernel_size': hp.choice('kernel_size', [2, 3, 4])
}

print("Loading data...")
file_path = "./Model/Data/multiple species.csv"
if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' does not exist.")
    exit(1)

file_seed = hash(os.path.basename(file_path)) % 2 ** 32
random.seed(file_seed)
np.random.seed(file_seed)
tf.random.set_seed(file_seed)

print(f"Loading data from {file_path}...")
X = pd.read_csv(file_path, header=None).values
if X.shape[1] < 1626:
    print("Error: Insufficient features. Expected at least 1626 columns.")
    exit(1)

X = X.reshape((X.shape[0], X.shape[1], 1))
num_samples = X.shape[0]
y = np.array([1] * (num_samples // 2) + [0] * (num_samples // 2))

train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.2, random_state=file_seed)
train_X, train_y = shuffleData(train_X, train_y, seed=file_seed)
valid_X, valid_y = shuffleData(valid_X, valid_y, seed=file_seed)


def objective(params):
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    learning_rate = params['learning_rate']
    dropout_rate = params['dropout_rate']
    batch_size = int(params['batch_size'])
    epochs = params['epochs']

    filters_onehot = [params['filters_onehot_1'], params['filters_onehot_2'], params['filters_onehot_3']]
    filters_vec_2 = [params['filters_vec_2_1'], params['filters_vec_2_2'], params['filters_vec_2_3']]
    filters_shape = [params['filters_shape_1'], params['filters_shape_2'], params['filters_shape_3']]
    kernel_size = params['kernel_size']

    model = CropGPA2_0(filters_onehot, kernel_size, filters_vec_2, filters_shape, dropout_rate, seed=RANDOM_SEED)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    history = model.fit(
        train_X, train_y,
        validation_data=(valid_X, valid_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=0
    )

    val_loss = min(history.history['val_loss'])
    tf.keras.backend.clear_session()
    del model
    return {'loss': val_loss, 'status': STATUS_OK}


trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=200,
            trials=trials, rstate=np.random.RandomState(file_seed))
best_params = space_eval(space, best)
print("Best hyperparameters:", best_params)

full_X = np.concatenate([train_X, valid_X], axis=0)
full_y = np.concatenate([train_y, valid_y], axis=0)
full_X, full_y = shuffleData(full_X, full_y, seed=file_seed)

best_learning_rate = best_params['learning_rate']
best_dropout_rate = best_params['dropout_rate']
best_batch_size = best_params['batch_size']
best_epochs = best_params['epochs']
best_filters_onehot = [best_params[f'filters_onehot_{i + 1}'] for i in range(3)]
best_filters_vec_2 = [best_params[f'filters_vec_2_{i + 1}'] for i in range(3)]
best_filters_shape = [best_params[f'filters_shape_{i + 1}'] for i in range(3)]
best_kernel_size = best_params['kernel_size']

output_file_name = os.path.splitext(os.path.basename(file_path))[0]
final_model = CropGPA2_0(best_filters_onehot, best_kernel_size, best_filters_vec_2, best_filters_shape,
                         best_dropout_rate, seed=file_seed)
optimizer = tf.keras.optimizers.Adam(learning_rate=best_learning_rate)
final_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

model_check = ModelCheckpoint(
    filepath=f"./{output_file_name}_Pre-model.h5",
    monitor='loss',
    save_best_only=True,
    save_weights_only=False
)

final_history = final_model.fit(
    full_X, full_y,
    epochs=best_epochs,
    batch_size=best_batch_size,
    callbacks=[
        model_check,
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    ],
    verbose=1
)

best_train_loss = min(final_history.history['loss'])
best_params_dict = {
    'file': os.path.basename(file_path),
    'learning_rate': best_learning_rate,
    'dropout_rate': best_dropout_rate,
    'batch_size': best_batch_size,
    'epochs': best_epochs,
    'final_train_epochs_used': final_history.epoch[-1] + 1,
    'best_train_loss': best_train_loss,
    'filters_onehot_1': best_filters_onehot[0],
    'filters_onehot_2': best_filters_onehot[1],
    'filters_onehot_3': best_filters_onehot[2],
    'filters_vec_2_1': best_filters_vec_2[0],
    'filters_vec_2_2': best_filters_vec_2[1],
    'filters_vec_2_3': best_filters_vec_2[2],
    'filters_shape_1': best_filters_shape[0],
    'filters_shape_2': best_filters_shape[1],
    'filters_shape_3': best_filters_shape[2],
    'kernel_size': best_kernel_size,
    'model_save_path': f"./{output_file_name}_Pre-model.h5"
}

output_csv_path = f"./{output_file_name}_best_params.csv"
pd.DataFrame([best_params_dict]).to_csv(output_csv_path, index=False)

print(f"Best Train Loss: {best_train_loss}, Final Model saved to ./{output_file_name}_Pre-model.h5")

del final_model, trials, full_X, full_y
gc.collect()
tf.keras.backend.clear_session()

print("File processed successfully. Final model saved.")
