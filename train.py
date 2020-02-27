from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout

with open('OnionOrNot.csv', 'r') as f:
    data = f.readlines()

# Prepare X, y data
X_data = []
y_data = []
i=1
for d in data:
    x = d.strip()[:-2]
    y = int(d.strip()[-1])
    X_data.append(x)
    y_data.append(y)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.20, random_state=42)
num_words = 

model = models.Sequential()
model.add(Embedding(
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))
opt = tf.keras.optimizers.Adam(0.0001)
