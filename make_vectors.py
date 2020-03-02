from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from tqdm import tqdm

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

# Save y data
with open('y_train', 'wb') as f:
    pickle.dump(y_train, f)
with open('y_test', 'wb') as f:
    pickle.dump(y_test, f)

unique_words = []
for x in X_train:
    unique_words.extend(x.split())

# Remove duplicates
unique_words = list(set(unique_words))
num_words = len(unique_words)
print(num_words)

# Encode strings as integers
X_train_encoded = []
for x in tqdm(X_train):
    headline = []
    for word in x.split():
        headline.append(unique_words.index(word))
    headline = np.asarray(headline)
    X_train_encoded.append(headline)
OOV = num_words
X_test_encoded = []
for x in tqdm(X_test):
    headline = []
    for word in x.split():
        try:
            headline.append(unique_words.index(word))
        except ValueError:
            headline.append(47029)
            OOV += 1
    headline = np.asarray(headline)
    X_test_encoded.append(headline)
with open('X_train_encoded', 'wb') as f:
    pickle.dump(X_train_encoded, f)
with open('X_test_encoded', 'wb') as f:
    pickle.dump(X_test_encoded, f)
