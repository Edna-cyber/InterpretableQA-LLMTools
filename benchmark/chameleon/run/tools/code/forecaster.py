import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

def forecast(model_name, previous_data): # return a list of forecasts
    """
    Forecasts the next data point given previous data
    """
    if model_name=="linear_regression":
        model = LinearRegression()
        model.fit(np.arange(len(previous_data)).reshape(-1,1), previous_data)
        preds = model.predict(np.array([len(previous_data)]).reshape(-1,1))
    if model_name=="ARIMA":
        # differencing -> make data stationary
        model = ARIMA(previous_data, order=(1,1,1))
        preds = model.fit().get_forecast(steps=1).predicted_mean.tolist()
    return preds

if __name__ == "__main__":
    previous_data = [0.628499, 0.612048, 0.620626, 0.655844, 0.552381, 0.539078, 0.507830, 0.516049, 0.482143, 0.483444, 0.490476, 0.425781, 0.417004, 0.430966, 0.400474, 0.379142, 0.375000, 0.333333, 0.291841, 0.252209, 0.238431, 0.214214, 0.185701, 0.189482]
    answer = forecast("linear_regression", previous_data)
    # answer = forecast("ARIMA", previous_data)
    print(answer)