import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tqdm import tqdm
import pickle
import numpy as np

# Load data.
with open('vectors/X_train_encoded', 'rb') as f:
    X_train = np.asarray(pickle.load(f))
with open('vectors/X_test_encoded', 'rb') as f:
    X_test = np.asarray(pickle.load(f))
with open('vectors/y_train', 'rb') as f:
    y_train = np.asarray(pickle.load(f))
with open('vectors/y_test', 'rb') as f:
    y_test = np.asarray(pickle.load(f))

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

X_train = keras.preprocessing.sequence.pad_sequences(X_train, value=38517, padding='post', maxlen=62, truncating='post')
X_test  = keras.preprocessing.sequence.pad_sequences(X_test,  value=38517, padding='post', maxlen=62, truncating='post')
print(X_train.shape)
print(y_train.shape)
num_words = 38518

model = models.Sequential()
model.add(Embedding(num_words, 5, input_length=62))
model.add(GlobalAveragePooling1D())
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
#opt = tf.keras.optimizers.Adam(0.0001)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=100, epochs=5000, callbacks=[es], validation_data=(X_test, y_test))
#history = model.fit(X_train, y_train, batch_size=1, epochs=5000)

model.save("model.h5")
