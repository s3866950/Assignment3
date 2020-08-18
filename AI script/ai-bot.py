import abc
import threading
import time
import pandas as pd
import numpy as np
from keras.layers import Dense
from keras.models import Sequential, model_from_json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from alpaca_trade_api import REST

class AlpacaPaperSocket(REST):
    def __init__(self):
        # Note: this is a paper account so it doesn't matter if it runs (doesn't use real money)
        super().__init__(key_id='PK1OKQT2WJATM96KFC2Y', secret_key='li3W1/K5oQPRw8GUmyi7/tLJWBHUUnwYawqWfwHj',base_url='https://paper-api.alpaca.markets', api_version='v2')


class TradingSystem(abc.ABC):
    def __init__(self, api, symbol, time_frame, system_id, system_label):
        self.api = api
        self.symbol = symbol
        self.time_frame = time_frame
        self.system_id = system_id
        self.system_label = system_label
        thread = threading.Thread(target=self.system_loop)
        thread.start()

    @abc.abstractmethod
    def place_buy_order(self):
        pass

    @abc.abstractmethod
    def place_sell_order(self):
        pass

    @abc.abstractmethod
    def system_loop(self):
        pass


class AIPMDevelopment:
    def __init__(self):

        data = pd.read_csv("C:\\Users\Cyqui\Desktop\python\IBM.csv")
        x = data['Delta Close']
        y = data.drop(['Delta Close'], axis=1)

        x_train, x_test, y_train, y_test = train_test_split(x, y)

        network = Sequential()

        network.add(Dense(1, input_shape=(1,), activation='tanh'))
        network.add(Dense(3, activation='tanh'))
        network.add(Dense(3, activation='tanh'))
        network.add(Dense(3, activation='tanh'))
        network.add(Dense(1, activation='tanh'))

        network.compile(optimizer='rmsprop', loss='hinge', metrics=['accuracy'])

        network.fit(x_train.values, y_train.values, epochs=100)

        y_pred = network.predict(x_test.values)
        y_pred = np.around(y_pred, 0)
        print("AIPMDevelopment Classification Report")
        print(classification_report(y_test, y_pred))

        model = network.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model)

        network.save_weights("weights.h5")

# uncomment line belowto print the training info for the AIPMDevelopment class
# AIPMDevelopment()

class PortfolioManagementModel:
    def __init__(self):
        data = pd.read_csv("C:\\Users\Cyqui\Desktop\python\IBM.csv")
        x = data['Delta Close']
        y = data.drop(['Delta Close'], axis=1)

        json_file = open('model.json', 'r')
        json = json_file.read()
        json_file.close()
        self.network = model_from_json(json)

        self.network.load_weights("weights.h5")

        y_pred = self.network.predict(x.values)
        y_pred = np.around(y_pred, 0)

        print("PortfolioManagementModel Classification Report:")
        print(classification_report(y, y_pred))

class PortfolioManagementSystem(TradingSystem):
    def __init__(self):
        super().__init__(AlpacaPaperSocket(), 'IBM', 86400, 1, 'AI_PM')
        self.AI = PortfolioManagementModel()

    def place_buy_order(self):
        self.api.submit_order(symbol='IBM', qty=1, side='buy', type='market', time_in_force='day',)

    def place_sell_order(self):
        self.api.submit_order(symbol='IBM', qty=1, side='sell', type='market', time_in_force='day',)

    def system_loop(self):
        this_weeks_close = 0
        last_weeks_close = 0
        delta = 0
        day_count = 0

        i = 0

        while(True):
            i += 1

            # this is just to make the buy/sell code work in the example code
            day_count += 1

            # wait 3 seconds between iterations. This would normally be much higher.
            time.sleep(3)

            print("Loop Iteration: ", i)

            data_req = self.api.get_barset('IBM', timeframe='1D', limit=1).df

            # print the bar set (time, open, high, low, close, volume)
            # print(data_req)

            x = pd.DataFrame(
                data=[[
                data_req['IBM']['close'][0]]],
                columns='Close'.split()
            )

            # print the selected data frame info
            # print("\nData frame: \n", x)
            
            if (day_count == 7):
                day_count = 0
                last_weeks_close = this_weeks_close
                this_weeks_close = x['Close']
                delta = this_weeks_close - last_weeks_close

                result = np.around(self.AI.network.predict([delta]))

                if result <= -0.5:
                    self.place_sell_order()
                    print("A sell was placed")
                elif result >= 0.5:
                    self.place_buy_order()
                    print("A buy was placed")


PortfolioManagementSystem()