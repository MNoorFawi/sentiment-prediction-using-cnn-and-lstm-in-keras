import pandas as pd

reviews = pd.read_csv("IMDB Dataset.csv")
reviews.head()
reviews.shape
# (50000, 2)

import re
from sklearn.feature_extraction import text

stop_words = text.ENGLISH_STOP_WORDS

def clean_review(review, stopwords):
    html_tag = re.compile('<.*?>')
    cleaned_review = re.sub(html_tag, "", review).split()
    cleaned_review = [i for i in cleaned_review if i not in stopwords]
    return " ".join(cleaned_review)

## before cleaning
text = reviews.review[0]
print(text[:200])
# One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me abo

## after cleaning
cleaned_text = clean_review(text, stop_words)
print(cleaned_text[:200])
# One reviewers mentioned watching just 1 Oz episode you'll hooked. They right, exactly happened me.The thing struck Oz brutality unflinching scenes violence, set right word GO. Trust me, faint hearted

## cleaning the review column
reviews["cleaned_review"] = reviews["review"].apply(lambda x: clean_review(x, stop_words))

from keras.preprocessing.text import Tokenizer

## maximum words to keep based on frequency 
max_features = 5000
## replace out-of-vocab words with this
oov = "OOV"
tokenizer = Tokenizer(num_words = max_features, oov_token = oov)
tokenizer.fit_on_texts(reviews["cleaned_review"])
## convert text into integers
tokenized = tokenizer.texts_to_sequences(reviews["cleaned_review"])

from sklearn.preprocessing import LabelEncoder

def sentiment_encode(df, column, le):
    le.fit(df[column])
    sentiment_le = le.transform(df[column])
    return sentiment_le, le

le = LabelEncoder()
sentiment_le, le = sentiment_encode(reviews, "sentiment", le)
print(len(le.classes_))
# 2
le.classes_
# array(['negative', 'positive'], dtype=object)

from keras.preprocessing import sequence

max_len = 500
Xtrain = sequence.pad_sequences(tokenized, maxlen = max_len)

from sklearn.model_selection import train_test_split

## we will do the splitting using a random state to ensure same splitting every time
X_train, X_test, y_train, y_test = train_test_split(Xtrain, sentiment_le, 
                                                    test_size = .5,
                                                    random_state = 13)
                                                    
## importing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv1D, MaxPooling1D

## model parameters
vocab_size = max_features #5000
embedding_dims = 128 # dimensions to which text will be represented
num_epochs = 3
noutput = len(le.classes_) #2 (binary)

## model
model = Sequential()
# embedding layer (vocab_size is the total number of words in data,
# then the embedding dimensions we specified, then the maximum length of one review)
model.add(Embedding(vocab_size, embedding_dims, input_length = max_len))
# CNN
model.add(Conv1D(128, kernel_size = 4, input_shape = (vocab_size, embedding_dims),
                 activation = "relu"))
# max pooling layer
model.add(MaxPooling1D(pool_size = 3))
# bidirectional LSTM
model.add(Bidirectional(LSTM(64, return_sequences = True)))
# LSTM and droput
model.add(LSTM(32, recurrent_dropout = 0.4))
model.add(Dropout(0.2))
# 1 neuron output layer and sigmoid activation (binary 0 or 1)
model.add(Dense(noutput - 1, activation = "sigmoid"))
# model summary and layout
model.summary()

# adam optimizer and binary crossentropy
model.compile(loss = "binary_crossentropy", metrics = ["accuracy"],
              optimizer = "adam")

model.fit(X_train, y_train, epochs = num_epochs,
          batch_size = 32,
          validation_data = (X_test[:1000], y_test[:1000]),
          verbose = 1)

results = model.evaluate(X_test[1000:], y_test[1000:])
# 750/750 [==============================] - 51s 65ms/step - loss: 0.3550 - accuracy: 0.8637
print("test loss: %.2f" % results[0])
# test loss: 0.36
print("test accuracy: %.2f%%" % (results[1] * 100))
# test accuracy: 86.37%