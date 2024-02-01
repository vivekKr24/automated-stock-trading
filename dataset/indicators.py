import pandas as pd


def simple_moving_average(time_series_data, col_name='SMA'):
    time_series_data[col_name] = time_series_data['Close'].rolling(window=14).mean()
    return time_series_data


def relative_strength_index(time_series_data, window_size=14, col_name='RSI'):
    time_series_data['price_change'] = time_series_data['Close'].diff()

    average_gain = time_series_data['price_change'].apply(lambda x: x if x > 0 else 0).rolling(
        window=window_size).mean()
    average_loss = -time_series_data['price_change'].apply(lambda x: x if x < 0 else 0).rolling(
        window=window_size).mean()

    relative_strength = average_gain / average_loss

    rsi = 100 - (100 / (1 + relative_strength))

    time_series_data[col_name] = rsi
    time_series_data = time_series_data.drop('price_change', axis=1)
    return time_series_data


def exponential_moving_average(time_series_data, alpha=0.2, col_name='EMA'):
    time_series_data[col_name] = time_series_data['Close'].ewm(alpha=alpha, adjust=False).mean()
    return time_series_data


def stochastic_oscillator(time_series_data, periods=14, col_name='%K'):
    time_series_data['lowest_low'] = time_series_data['Close'].rolling(window=periods).min()
    time_series_data['highest_high'] = time_series_data['Close'].rolling(window=periods).max()
    time_series_data['%K'] = ((time_series_data['Close'] - time_series_data['lowest_low']) / (
            time_series_data['highest_high'] - time_series_data['lowest_low'])) * 100

    time_series_data[col_name] = time_series_data['%K'].rolling(window=3).mean()
    time_series_data: pd.DataFrame = time_series_data.drop(['lowest_low', 'highest_high'], axis=1)
    return time_series_data


def MACD(time_series_data, fast_window=12, slow_window=26, col_name='MACD'):
    time_series_data['EMA_fast'] = time_series_data['Close'].ewm(span=fast_window, adjust=False).mean()
    time_series_data['EMA_slow'] = time_series_data['Close'].ewm(span=slow_window, adjust=False).mean()

    time_series_data['MACD_line'] = time_series_data['EMA_fast'] - time_series_data['EMA_slow']

    signal_window = 9
    time_series_data['Signal_line'] = time_series_data['MACD_line'].ewm(span=signal_window, adjust=False).mean()

    time_series_data[col_name] = time_series_data['MACD_line'] - time_series_data['Signal_line']

    time_series_data = time_series_data.drop(['MACD_line', 'Signal_line', 'EMA_fast', 'EMA_slow'], axis=1)
    return time_series_data


def accumulation_distribution(time_series_data, col_name='AD'):
    time_series_data['MF_multiplier'] = ((time_series_data['Close'] - time_series_data['Low']) - (
            time_series_data['High'] - time_series_data['Close'])) / (
                                                time_series_data['High'] - time_series_data['Low'])

    time_series_data['MF_volume'] = time_series_data['MF_multiplier'] * time_series_data['Volume']

    time_series_data[col_name] = time_series_data['MF_volume'].cumsum()

    time_series_data = time_series_data.drop(['MF_volume', 'MF_multiplier'], axis=1)
    return time_series_data


def obv(time_series_data, col_name='OBV'):
    time_series_data['price_change'] = time_series_data['Close'].diff()

    time_series_data['direction'] = pd.cut(time_series_data['price_change'], bins=[float('-inf'), 0, float('inf')],
                                           labels=['down', 'up'])
    time_series_data['OBV'] = time_series_data['Volume'] * (time_series_data['direction'] == 'up') - time_series_data[
        'Volume'] * (time_series_data['direction'] == 'down')

    time_series_data[col_name] = time_series_data['OBV'].cumsum()
    time_series_data = time_series_data.drop(['price_change', 'direction'], axis=1)
    return time_series_data


def roc(time_series_data, periods=3, col_name='ROC'):
    # Calculate the Price Rate of Change (ROC)
    time_series_data[col_name] = time_series_data['Close'].pct_change(periods=periods) * 100
    return time_series_data


def williams_R(time_series_data, lookback_period=14, col_name='Williams R'):
    time_series_data['highest_high'] = time_series_data['High'].rolling(window=lookback_period).max()
    time_series_data['lowest_low'] = time_series_data['Low'].rolling(window=lookback_period).min()

    time_series_data[col_name] = (time_series_data['highest_high'] - time_series_data['Close']) / (
            time_series_data['highest_high'] - time_series_data['lowest_low']) * -100
    time_series_data = time_series_data.drop(['lowest_low', 'highest_high'], axis=1)
    return time_series_data


def disparity_index(time_series_data, lookback_period=3, col_name='Disparity_Index'):
    time_series_data['MA'] = time_series_data['Close'].rolling(window=lookback_period).mean()

    time_series_data[col_name] = ((time_series_data['Close'] - time_series_data['MA']) / time_series_data['MA']
                                  ) * 100
    time_series_data = time_series_data.drop(['MA'], axis=1)
    return time_series_data


def get_default_indicators():
    return {
        'SMA': simple_moving_average,
        'EMA': exponential_moving_average,
        'MACD': MACD,
        'RSI': relative_strength_index,
        '%K': stochastic_oscillator,
        'AD': accumulation_distribution,
        'OBV': obv,
        'ROC': roc,
        '%R': williams_R,
        'Disparity_Index': disparity_index
    }
