import os
import random
import numpy as np
import pandas as pd
from keras.layers import Conv1D, Dense, Dropout, Flatten, Input, Concatenate, Lambda, MaxPooling1D, BatchNormalization
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l2
import tensorflow as tf
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval

# Global random seed
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Detected GPUs: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled memory growth for GPUs.")
    except RuntimeError as e:
        print(f"GPU setup failed: {e}")
else:
    print("No GPUs detected.")

def shuffle_data(X, y, seed=RANDOM_SEED):
    index = np.arange(len(X))
    np.random.seed(seed)
    np.random.shuffle(index)
    return X[index], y[index]

def CropGPA2_0(filters_onehot, kernel_size, filters_vec_2, filters_shape, dropout_rate, seed):
    tf.random.set_seed(seed)
    inputs = Input(shape=(1623, 1))

    # Split features
    onehot = Lambda(lambda x: x[:, :164, :])(inputs)
    vec_2 = Lambda(lambda x: x[:, 164:328, :])(inputs)
    bert = Lambda(lambda x: x[:, 328:1096, :])(inputs)
    shape = Lambda(lambda x: x[:, 1096:, :])(inputs)

    # onehot module
    onehot = Conv1D(filters=filters_onehot[0], kernel_size=kernel_size, strides=1, padding="same",
                    kernel_regularizer=l2(0.001))(onehot)
    onehot = Conv1D(filters=filters_onehot[1], kernel_size=kernel_size, strides=1, padding="same",
                    kernel_regularizer=l2(0.001))(onehot)
    onehot = MaxPooling1D(pool_size=3, strides=1, padding='valid')(onehot)
    onehot = Conv1D(filters=filters_onehot[2], kernel_size=kernel_size, strides=1, padding="same",
                    kernel_regularizer=l2(0.001))(onehot)
    onehot = Flatten()(onehot)

    # vec_2 module
    vec_2 = Conv1D(filters=filters_vec_2[0], kernel_size=kernel_size, strides=1, padding="same",
                   kernel_regularizer=l2(0.001))(vec_2)
    vec_2 = Conv1D(filters=filters_vec_2[1], kernel_size=kernel_size, strides=1, padding="same",
                   kernel_regularizer=l2(0.001))(vec_2)
    vec_2 = MaxPooling1D(pool_size=3, strides=1, padding='valid')(vec_2)
    vec_2 = Conv1D(filters=filters_vec_2[2], kernel_size=kernel_size, strides=1, padding="same",
                   kernel_regularizer=l2(0.001))(vec_2)
    vec_2 = Flatten()(vec_2)

    # bert module
    bert = Flatten()(bert)
    bert = Dense(768, activation='relu', kernel_regularizer=l2(0.001))(bert)
    bert = Dropout(dropout_rate)(bert)
    bert = Dense(1024, activation='relu', kernel_regularizer=l2(0.001))(bert)
    bert = Dropout(dropout_rate)(bert)
    bert = Flatten()(bert)

    # shape module
    shape = Conv1D(filters=filters_shape[0], kernel_size=kernel_size, strides=1, padding="same",
                   kernel_regularizer=l2(0.001))(shape)
    shape = Conv1D(filters=filters_shape[1], kernel_size=kernel_size, strides=1, padding="same",
                   kernel_regularizer=l2(0.001))(shape)
    shape = MaxPooling1D(pool_size=4, strides=1, padding='valid')(shape)
    shape = Conv1D(filters=filters_shape[2], kernel_size=kernel_size, strides=1, padding="same",
                   kernel_regularizer=l2(0.001))(shape)
    shape = Flatten()(shape)

    # Concatenate
    concatenated = Concatenate(axis=-1)([onehot, vec_2, bert, shape])
    flattened = Flatten()(concatenated)

    # Fully connected layers
    dense1 = Dense(2048, kernel_initializer='glorot_normal', activation='relu', name='dense1',
                   kernel_regularizer=l2(0.001))(flattened)
    dense2 = Dense(256, kernel_initializer='glorot_normal', activation='relu', name='dense2',
                   kernel_regularizer=l2(0.001))(dense1)
    dense3 = Dense(32, activation='relu', name='dense3', kernel_regularizer=l2(0.001))(dense2)

    # Output layer
    output = Dense(1, activation='sigmoid', name='output')(dense3)

    model = Model(inputs=inputs, outputs=output)
    print(model.summary())
    return model

def objective(params, file_seed):
    tf.random.set_seed(file_seed)
    np.random.seed(file_seed)

    learning_rate = params['learning_rate']
    dropout_rate = params['dropout_rate']
    batch_size = int(params['batch_size'])
    epochs = params['epochs']

    filters_onehot = [params['filters_onehot_1'], params['filters_onehot_2'], params['filters_onehot_3']]
    filters_vec_2 = [params['filters_vec_2_1'], params['filters_vec_2_2'], params['filters_vec_2_3']]
    filters_shape = [params['filters_shape_1'], params['filters_shape_2'], params['filters_shape_3']]
    kernel_size = params['kernel_size']

    model = CropGPA2_0(filters_onehot, kernel_size, filters_vec_2, filters_shape, dropout_rate, seed=file_seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    history = model.fit(
        train_X, train_y,
        validation_data=(valid_X, valid_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)],
        verbose=1
    )

    val_loss = history.history['val_loss'][-1]
    return {'loss': val_loss, 'status': STATUS_OK}

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
folder_path = "./Model/Data/multiple species.csv"
if not os.path.exists(folder_path):
    print(f"Error: Folder '{folder_path}' does not exist.")
    exit(1)

data_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
if not data_files:
    print("No CSV files found in the folder.")
    exit(1)

for data_file in data_files:
    file_seed = hash(data_file) % (2 ** 32)
    random.seed(file_seed)
    np.random.seed(file_seed)
    tf.random.set_seed(file_seed)

    data_path = os.path.join(folder_path, data_file)
    print(f"Loading data from {data_path}...")

    X = pd.read_csv(data_path, header=None).values
    print(f"Loaded data shape: {X.shape}")

    if X.shape[1] < 1623:
        print("Error: Insufficient features. Expected at least 1623 columns.")
        continue

    X = X.reshape((X.shape[0], X.shape[1], 1))
    num_samples = X.shape[0]
    y = np.array([1] * (num_samples // 2) + [0] * (num_samples // 2))

    train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.2, random_state=file_seed)

    train_X, train_y = shuffle_data(train_X, train_y, seed=file_seed)
    valid_X, valid_y = shuffle_data(valid_X, valid_y, seed=file_seed)

    trials = Trials()
    best = fmin(
        fn=lambda params: objective(params, file_seed),
        space=space,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials,
        rstate=np.random.RandomState(file_seed)
    )

    best_params = space_eval(space, best)
    print("Best hyperparameters:", best_params)

    br = best_params['learning_rate']
    dr = best_params['dropout_rate']
    bs = best_params['batch_size']
    ep = best_params['epochs']
    fo = [best_params[f'filters_onehot_{i + 1}'] for i in range(3)]
    fv = [best_params[f'filters_vec_2_{i + 1}'] for i in range(3)]
    fs = [best_params[f'filters_shape_{i + 1}'] for i in range(3)]
    ks = best_params['kernel_size']

    print(f"Using learning rate: {br}")
    print(f"Using dropout rate: {dr}")
    print(f"Using batch size: {bs}")
    print(f"Using epochs: {ep}")
    print(f"Using filters_onehot: {fo}")
    print(f"Using filters_vec_2: {fv}")
    print(f"Using filters_shape: {fs}")
    print(f"Using kernel size: {ks}")

    output_name = os.path.splitext(data_file)[0]

    model = CropGPA2_0(fo, ks, fv, fs, dr, seed=file_seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=br)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])

    os.makedirs("./checkpoints", exist_ok=True)
    ckpt_path = os.path.join("./checkpoints", f"Pre-model.h5")
    model_check = ModelCheckpoint(filepath=ckpt_path, monitor='val_binary_accuracy', save_best_only=True)

    history = model.fit(
        train_X, train_y,
        validation_data=(valid_X, valid_y),
        epochs=ep,
        batch_size=bs,
        callbacks=[model_check],
        verbose=1
    )

    best_val_loss = min(history.history['val_loss'])
    best_val_acc = max(history.history['val_binary_accuracy'])
    params_dict = {
        'file': data_file,
        'learning_rate': br,
        'dropout_rate': dr,
        'batch_size': bs,
        'epochs': ep,
        'filters_onehot_1': fo[0],
        'filters_onehot_2': fo[1],
        'filters_onehot_3': fo[2],
        'filters_vec_2_1': fv[0],
        'filters_vec_2_2': fv[1],
        'filters_vec_2_3': fv[2],
        'filters_shape_1': fs[0],
        'filters_shape_2': fs[1],
        'filters_shape_3': fs[2],
        'kernel_size': ks,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'model_save_path': f"{output_name}_model.h5"
    }

    import gc
    del model
    gc.collect()
    tf.keras.backend.clear_session()

print("Training completed.")