from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.layers import Embedding, Dense, Bidirectional, Dropout, CuDNNGRU,Conv1D,BatchNormalization,Activation,MaxPool1D,Deconv2D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pickle
from Attention import Attention
max_len = 80
num_words = 20000

train_list = [i.strip() for i in open('DATA/train.txt', 'r', encoding='utf8')]
tokenizer = joblib.load('tokenizer_final.model')
# tokenizer = Tokenizer(num_words=num_words)
# tokenizer.fit_on_texts(train_list)
# joblib.dump(tokenizer,'tokenizer_final.model')
train_list = tokenizer.texts_to_sequences(train_list)
x_train = pad_sequences(train_list, max_len)
one_hot = {"0":[0,1],"1":[1,0]}
y_train = np.array([one_hot[i.replace("\n","")] for i in open('DATA/label.txt', 'r', encoding='utf8')])
train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size=0.1, random_state=10)

# with open('data.pkl', 'wb') as f:
#     pickle.dump(x_train, f)
#     pickle.dump(y_train, f)
#     pickle.dump(x_test, f)
#     pickle.dump(y_test, f)

# with open('data.pkl', 'rb') as f:
#     x_train = pickle.load(f)
#     y_train = pickle.load(f)
#     x_test = pickle.load(f)
#     y_test = pickle.load(f)


model = Sequential([
    Embedding(num_words + 1, 64, input_shape=(max_len,)),
    Conv1D(64, 3, padding='same'),
    BatchNormalization(),
    Activation('relu'),
    # MaxPool1D(10),
    Bidirectional(CuDNNGRU(128, return_sequences=True), merge_mode='sum'),
    Attention(64),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr = 0.001), metrics=['acc'])
save_best = ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(patience=1, verbose=1, cooldown=1, factor=0.4)
early_stop = EarlyStopping(patience=3, verbose=1)
model.fit(train_x,train_y, batch_size=100, epochs=200, validation_data=(test_x, test_y), callbacks=[ save_best,early_stop,reduce_lr])
# i = 0
# count = 0
# for text, label in zip(train_list, label_list):
#     result = model.predict(np.expand_dims(text,0))
#     result = lb.inverse_transform(result)
#     if str(result[0]) == label:
#         i +=1
#     count += 1
#     print(count)
# print('i:{}, count:{}, acc: {}'.format(i, count, i / count))