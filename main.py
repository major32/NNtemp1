import matplotlib as mpl
import matplotlib.pyplot
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from keras.losses import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Activation, Dense, Dropout, CuDNNLSTM, LSTM, RNN, SimpleRNN, LeakyReLU, GRU, CuDNNGRU

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
scaler = MinMaxScaler(feature_range=(-1, 1))


def CreateNN():
    np.random.seed(42)
    test_size = 0.2
    univariate_past_history = 500
    univariate_future_target = 1
    #uni_data = pd.read_csv("E:/install/15mkline.csv")
    uni_data = pd.read_csv("E:/install/15mkline.csv").tail(50000).head(45000)
    #uni_data = pd.read_csv("E:/install/15mkline.csv").tail(116000)
    uni_data = uni_data.set_index('<DATE>')
    uni_data.index = pd.to_datetime(uni_data.index)
    uni_data.drop(["<OPEN>", "<LOW>", "<HIGH>", "<VOL>"], axis='columns', inplace=True)
    #uni_data.plot(subplots=False)

    TRAIN_SPLIT = len(uni_data) - int(test_size * len(uni_data))
    uni_data = uni_data.values
    print(uni_data)

    scaled_data = scaler.fit_transform(uni_data)

    # matplotlib.pyplot.show()
    # print(scaled_data)
    # print(uni_data)


    x_train_uni, y_train_uni = univariate_data(scaled_data, 0, TRAIN_SPLIT, univariate_past_history,
                                               univariate_future_target)
    x_val_uni, y_val_uni = univariate_data(scaled_data, TRAIN_SPLIT, None, univariate_past_history,
                                           univariate_future_target)

    #show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')
    matplotlib.pyplot.show()
    #print('Single window of past history')
    #print(x_train_uni[0].head(10))
    #print('\n Target price to predict')
    #print(y_train_uni[0])

    # print(x_val_uni)
    # show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0, 'Среднее значение (MA)').show()

    BATCH_SIZE = 32
    BUFFER_SIZE = 10000 #50000

    train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
    train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
    val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

    """simple_lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(250, input_shape=x_train_uni.shape[-2:]),
        tf.keras.layers.Dense(1)
    ])
    """
    simple_lstm_model = build_lstm_model(input_data=x_train_uni.shape[-2:])
    simple_lstm_model.compile(optimizer='adam', loss='mape', metrics=['mean_absolute_error'])
    # show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')
    EVALUATION_INTERVAL = 100  #100
    EPOCHS = 60 #20


    #Почистить папку с логами

    import os, shutil
    folder = 'E:/install/models/logs/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Если ошибка не уменьшается на протяжении указанного количества эпох, то процесс обучения прерывается и модель инициализируется
    # весами с самым низким показателем параметра "monitor"
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_mean_absolute_error',
        # указывается параметр, по которому осуществляется ранняя остановка. Обычно это функция потреть на валидационном наборе (val_loss)
        patience=5,  # количество эпох по истечении которых закончится обучение, если показатели не улучшатся
        mode='min',  # указывает, в какую сторону должна быть улучшена ошибка
        restore_best_weights=True
        # если параметр установлен в true, то по окончании обучения модель будет
        # инициализирована весами с самым низким показателем параметра "monitor"
    )

    # Сохраняет модель для дальнейшей загрузки
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="E:/install/models/",  # путь к папке, где будет сохранена модель
        monitor='val_mean_absolute_error',
        save_best_only=True,  # если параметр установлен в true, то сохраняется только лучшая модель
        mode='min'
    )

    # Сохраняет логи выполнения обучения, которые можно будет посмотреть в специальной среде TensorBoard
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir="E:/install/models/logs/",  # путь к папке где будут сохранены логи
    )

    simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                          steps_per_epoch=EVALUATION_INTERVAL,
                          validation_data=val_univariate, validation_steps=50, verbose=1, shuffle=True,
                          callbacks=[
                              early_stopping,
                              #model_checkpoint,
                              tensorboard
                          ]
                          )


    #print('val_univariate:  ', val_univariate)
    #print('Predict: ', simple_lstm_model.predict(val_univariate[0]))

    for x, y in val_univariate.take(3):
        predic =  simple_lstm_model.predict(x)[0]
        print("predic:",predic)
        plot = show_plot([x[0].numpy(), y[0].numpy(),
                          predic], 0, 'Simple LSTM model')
        #print("MAE: ", mean_absolute_error(simple_lstm_model.predict(x).numpy(), y[0].numpy()))
        plot.show()

    #simple_lstm_model.save("D:/Nerualsnapshot/temp.h5")

#loss='mse'
def build_lstm_model(input_data, output_size=1, neurons=500, neurons2=2499, activ_func='linear',
                     dropout=0.3, loss='mae', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(units=neurons, input_shape=input_data, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=neurons2, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=neurons2))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model


def create_time_steps(length):
    return list(range(-length, 0))


def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])  # Возврат -20 в -1 список
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10,
                     label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel('Time-Step')
    return plt


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)


def baseline(history):
    return np.mean(history)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #from tensorflow.python.client import device_lib
    #print(device_lib.list_local_devices())
    CreateNN()

