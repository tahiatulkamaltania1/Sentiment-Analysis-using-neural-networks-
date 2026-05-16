import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Dataset (1 = Positive Sentiment, 0 = Negative Sentiment)
sentences = [
    'I love this university course, it is amazing',
    'This is the worst project experience ever',
    'Highly recommended and very helpful content',
    'Waste of time and very boring lecture'
]
labels = np.array([1, 0, 1, 0])

# 2. Tokenize and Pad the Sentences
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=8, padding='post')

# 3. Create Neural Network Architecture with Embedding Layer
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(100, 8, input_length=8),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Sigmoid for Binary Classification (0 or 1)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(padded, labels, epochs=5, verbose=0)

print("Project 05: Sentiment Analysis Using Neural Networks")
print("-----------------------------------------------------")
print("Sentiment Model Architecture:")
model.summary()
print("\nText Embedding and Analysis Pipeline Executed Successfully!")
