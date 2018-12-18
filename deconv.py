import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Embedding, Dense, Conv1D, MaxPool1D,Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from keras.activations import relu
from keras import backend as K

max_len = 80
num_words = 20000

train_list = [i.strip() for i in open('DATA/train.txt', 'r', encoding='utf8')]
# tokenizer = joblib.load('tokenizer_final.model')
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(train_list)
joblib.dump(tokenizer,'tokenizer_final.model')
train_list = tokenizer.texts_to_sequences(train_list)
x_train = pad_sequences(train_list, max_len)
one_hot = {"0":[0,1],"1":[1,0]}
y_train = np.array([one_hot[i.replace("\n","")] for i in open('DATA/label.txt', 'r', encoding='utf8')])
train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size=0.1, random_state=10)

model = Sequential([
    Embedding(num_words + 1, 128, input_shape=(max_len,)),
    Conv1D(64,3,padding='same'),
    Conv1D(32,3,padding="same"),
    Conv1D(16,3,padding="same"),
    Conv1D(8,3,padding="same"),
    Conv1D(4,3,padding="same"),
    Conv1D(2,3,padding="same"),
    Conv1D(1,3,padding="same"),
    # Conv1D(2,3,padding="same"),
    # Conv1D(4,3,padding="same"),
    # Conv1D(8,3,padding="same"),
    # Conv1D(16,3,padding="same"),
    # Conv1D(32,3,padding="same"),
    # Conv1D(64,3,padding="same"),
    # Conv1D(128,3,padding="same"),
    Flatten(),
    Dense(2,activation="softmax")
])
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr = 0.001), metrics=['acc'])
save_best = ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)
reduce_lr = ReduceLROnPlateau(patience=1, verbose=1, cooldown=1, factor=0.4)
early_stop = EarlyStopping(patience=3, verbose=1)
model.fit(train_x,train_y, batch_size=100, epochs=20, validation_data=(test_x, test_y), callbacks=[ save_best,early_stop,reduce_lr])
a = model.get_weights()
get_1_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                              [model.layers[2].output])
b = get_1_layer_output([train_x[0:1], 0])[0]
pass
