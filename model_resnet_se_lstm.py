import os
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 13})

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)

from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score


def weighted_categorical_crossentropy(weights):

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


def resnet_se_block(inputs, num_filters, kernel_size, strides, ratio):      
    # 1D conv
    x = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)    
    x = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # se block
    se = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=-2, keepdims=True))(x) # equal to tf.keras.layers.GlobalAveragePooling1D
    se = tf.keras.layers.Dense(units=num_filters//ratio)(se)
    se = tf.keras.layers.Activation('relu')(se)
    se = tf.keras.layers.Dense(units=num_filters)(se)
    se = tf.keras.layers.Activation('sigmoid')(se)
    x = tf.keras.layers.multiply([x, se])
    
    # skip connection
    x_skip = tf.keras.layers.Conv1D(filters=num_filters, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    x_skip = tf.keras.layers.BatchNormalization()(x_skip)

    x = tf.keras.layers.Add()([x, x_skip])
    x = tf.keras.layers.Activation('relu')(x)
    # x = tf.keras.layers.MaxPool2D(pool_size=(2,1), strides=(2,1), padding='same', data_format = 'channels_first')(x)
    
    return x

def create_model(Fs = 100, n_classes=5, seq_length=15, summary=True):   
    x_input = tf.keras.Input(shape=(seq_length, 30*Fs, 1))  # (None, seq_length, 3000, 1)
    
    x = resnet_se_block(x_input, 32, 3, 1, 4)  # (None, seq_length, 3000, 32)    
    x = tf.keras.layers.MaxPool2D(pool_size=(4,1), strides=(4,1), padding='same', data_format = 'channels_first')(x)  # (None, seq_length, 750, 32)
    
    x = resnet_se_block(x, 64, 5, 1, 4)  # (None, seq_length, 750, 64) 
    x = tf.keras.layers.MaxPool2D(pool_size=(4,1), strides=(4,1), padding='same', data_format = 'channels_first')(x)  # (None, seq_length, 188, 64)
    
    x = resnet_se_block(x, 128, 7, 1, 4)  # (None, seq_length, 188, 128)
    x = tf.keras.layers.Lambda(lambda x: K.mean(x, axis=-2, keepdims=False))(x)  # (None, seq_length, 128), equal to tf.keras.layers.GlobalAveragePool
   
    x = tf.keras.layers.Dropout(rate=0.5)(x)  # (None, seq_length, 128)
        
    # LSTM
    x = tf.keras.layers.LSTM(units=64, dropout=0.5, activation='relu', return_sequences=True)(x)  # (None, seq_length, 64)
       
    # Classify
    x_out = tf.keras.layers.Dense(units=n_classes, activation='softmax')(x)  # (None, seq_length, 5)
    
    model = tf.keras.models.Model(x_input, x_out)
        
    model.compile(optimizer='adam', loss=weighted_categorical_crossentropy(np.array([1, 1.5, 1, 1, 1])), metrics=['accuracy'])  # np.array([1, 3, 1, 5, 3]), np.array([1.5, 2.5, 1, 1.5, 2.5])
    
    if summary:
        model.summary()
        tf.keras.utils.plot_model(model, show_shapes=True, dpi = 300, to_file='model.png')
        
    return model

## data preparation
data_path = 'data/sleepedf/sleep-cassette/eeg_fpz_cz'
# data_path = 'data/ISRUC_S1'

fnames = sorted(glob(os.path.join(data_path, '*.npz')))

X, y = [], []
for fname in fnames:
    samples = np.load(fname)
    X.append(samples['x'])
    y.append(samples['y'])

# one-hot encoding sleep stages    
temp_y = []
for i in range(len(y)):
    temp_ = []
    for j in range(len(y[i])):
        temp = np.zeros((5,))
        temp[y[i][j]] = 1.
        temp_.append(temp)
    temp_y.append(np.array(temp_))
y = temp_y    

# make sequence data
seq_length = 15

X_seq, y_seq = [], []
for i in range(len(X)):
    for j in range(0, len(X[i]), seq_length): # discard last short sequence
        if j+seq_length < len(X[i]):
            X_seq.append(np.array(X[i][j:j+seq_length]))
            y_seq.append(np.array(y[i][j:j+seq_length]))
            
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(X_seq, y_seq, test_size=0.1, random_state=42)
X_seq_train, X_seq_val, y_seq_train, y_seq_val = train_test_split(X_seq_train, y_seq_train, test_size=0.1, random_state=42)

X_seq_train = np.expand_dims(X_seq_train, -1)
X_seq_val = np.expand_dims(X_seq_val, -1)
X_seq_test = np.expand_dims(X_seq_test, -1)

## model training
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='model', monitor='val_loss', verbose=1, save_best_only=True)
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)
redonplat = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1)
csv_logger = tf.keras.callbacks.CSVLogger('log.csv', separator=',', append=True)
callbacks_list = [
    checkpoint,
    early,
    redonplat,
    csv_logger,
    ]
    
model = create_model(seq_length=seq_length)

hist = model.fit(X_seq_train, y_seq_train, batch_size=16, epochs=100, verbose=1,
                 validation_data=(X_seq_val, y_seq_val), callbacks=callbacks_list)

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='lower right')

plt.subplot(1,2,2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.suptitle('hist')
plt.savefig('hist.png')
plt.close()

model.save('model.h5')

## output
y_seq_pred = model.predict(X_seq_test, batch_size=1)

y_seq_pred_ = y_seq_pred.reshape(-1,5)
y_seq_test_ = y_seq_test.reshape(-1,5)
y_seq_pred_ = np.array([np.argmax(s) for s in y_seq_pred_])
y_seq_test_ = np.array([np.argmax(s) for s in y_seq_test_])

accuracy = accuracy_score(y_seq_test_, y_seq_pred_)
print('accuracy:', accuracy)

kappa = cohen_kappa_score(y_seq_test_, y_seq_pred_)
print('kappa:', kappa)

label = ['Wake', 'N1', 'N2', 'N3', 'REM']  

report = classification_report(y_true=y_seq_test_, y_pred=y_seq_pred_, target_names=label, output_dict=True)
print('report:', report) 
report = pd.DataFrame(report).transpose()
report.to_csv('report.csv', index= True)

cm = confusion_matrix(y_seq_test_, y_seq_pred_)
sns.heatmap(cm, square=True, annot=True, fmt='d', cmap='YlGnBu', xticklabels=label, yticklabels=label)
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.savefig('cm.png', bbox_inches='tight', dpi=300)
plt.close()

cm_norm = confusion_matrix(y_seq_test_, y_seq_pred_, normalize='true')
sns.heatmap(cm_norm, square=True, annot=True, cmap='YlGnBu', xticklabels=label, yticklabels=label)
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.savefig('cm_norm.png', bbox_inches='tight', dpi=300)
plt.close()

    