from tensorflow.keras.models import load_model
from tensorflow import keras
import pickle
import sys

with open('vocabulary', 'rb') as f:
    vocabulary = pickle.load(f)

s = sys.argv[1]

encoded = []
for word in s.split():
    print(word)
    try:
        encoded.append(vocabulary.index(word.lower()))
    except ValueError:
        encoded.append(38516)

encoded = keras.preprocessing.sequence.pad_sequences([encoded], value=38517, padding='post', maxlen=62, truncating='post')
 
model = load_model('model.h5')
print(model.predict(encoded))

