from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD

class FeedForwardNN():
    def __init__(self, x_train, y_train, hidden_units=20, activation='sigmoid', 
                loss_func='mean_squared_error', learning_rate=0.01, epochs=2000):
        self.x_train = x_train
        self.y_train = y_train
        self.hidden_units = hidden_units
        self.loss_func = loss_func
        self.activation = activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = None

    def fit(self):
        input_layer = Input(shape=self.x_train[0].shape)
        hidden_layer = Dense(units=self.hidden_units, activation=self.activation)(input_layer)
        output_layer = Dense(1)(hidden_layer)
        sgd = SGD(lr=self.learning_rate)
        self.model = Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer=sgd, loss=self.loss_func, metrics=['accuracy'])
        self.model.fit(self.x_train, self.y_train, len(self.x_train), self.epochs)

    def predict(self, data):
        return self.model.predict(data)