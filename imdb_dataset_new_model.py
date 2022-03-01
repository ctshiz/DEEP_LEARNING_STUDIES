from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt

#1. Loading the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

#print(max([max(sequence) for sequence in train_data]))

#decoding the reviews back to English words

#word_index is a dictionary mapping words to an integer index
word_index = imdb.get_word_index()
#reverses it, mapping integer indices to words
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]
)
#decodes the review. Note that the indices are offset by 3 because 0, 1, and 2 are reversed indices for "padding", "start of sequence", and "unknown"
decoded_review =' '.join(
    [reverse_word_index.get(i-3, '?') for i in train_data[0]]
)
#print(decoded_review)

#2. Encoding the integer sequences into a binary matrix
def vectorize_sequences(sequences, dimension=10000):
    #creates an all-zero matrix of shape(len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        #sets specific indices of results[i] to 1s
        results[i, sequence] = 1.
        return results

#vectorized training data
x_train = vectorize_sequences(train_data)
#vectorized test data
x_test = vectorize_sequences(test_data)
#vectorized labels
y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

#print(x_train[0])

#data is ready to be fed into a neural network

activation_list = ['relu', 'tanh']
loss_list = ['binary_crossentropy', 'mse']

results_list = []
#3. The model definition
model = models.Sequential()
for i in range(len(activation_list)):
    model.add(layers.Dense(16, activation=activation_list[i], input_shape=(10000,)))
    model.add(layers.Dense(16, activation=activation_list[i]))
    model.add(layers.Dense(1, activation='sigmoid'))

    for _ in range(len(loss_list)):
        #4. Compiling the model
        model.compile(optimizer='rmsprop',
                loss=loss_list[_],
                metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=4, batch_size = 512)
        results  = model.evaluate(x_test, y_test)

        #print(model.predict(x_test))
        results_list.append(model.predict(x_test))

print(results_list)