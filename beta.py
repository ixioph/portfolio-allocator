import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import sys

# TODO: Portfolio class that takes tickers and adds and removes them from main portfolio

class Beta():
# TODO: print function

    def __init__(self, security, start, end, benchmark='SPY', period='max'):
        self.period = period
        self.start = start
        self.end = end
        self.security = self.get_market_data(security)
        self.benchmark = self.get_market_data(benchmark)

        corr = self.corr()
        cov = self.covariance()
        var = self.variance()

        print(f'{security}/{benchmark}::')
        print(f'Correlation: {corr}\nCovariance: {cov}\nVariance: {var}\n')
        # calculate the beta
        self.beta = cov/var
        print('BETA: ', self.beta)
        return None

    def corr(self):
        s = self.security['Close']
        b = self.benchmark['Close']
        return np.corrcoef(s, b)[0, 1]

    def variance(self):
        b = self.benchmark['Close']
        return np.var(b)

    def covariance(self):
        s = self.security['Close']
        b = self.benchmark['Close']
        n = len(b.index)

        sum = 0
        for i in range(0, n):
            sum = (sum + (s[i] - s.mean()) * (b[i] - b.mean()))

        return sum / (n - 1)

    def get_market_data(self, symbol):
        try:
            md = yf.Ticker(symbol)
            df = md.history(period=self.period, start=self.start, end=self.end)
            return df
        except Exception as e:
            print(str(e))
            return -1


class Portfolio():
    benchmark = None
    portfolio = []
    breakdown = []
    sharpe = {}

    stdev = pd.DataFrame()

    daily_open = pd.DataFrame()
    daily_high = pd.DataFrame()
    daily_low = pd.DataFrame()
    daily_close = pd.DataFrame()
    daily_volume = pd.DataFrame()

    def __init__(self, portfolio=None, weights=None, benchmark='SPY', start='2019-3-16', end='2020-8-1'): #todo, default end today
        self.start = start
        self.end = end
        self.benchmark = benchmark
        data = self.pull_market_data(self.benchmark)
        self.initialize_charts(data)
        if portfolio != None:
            print('Portfolio:', portfolio)
            self.add_securities(portfolio)
            if weights != None:
                self.calculate_sharpe(weights)
            else: self.calculate_sharpe()
        #print(self.daily_close, ",/,", self.daily_close.std(axis=0))
        self.stdev = self.daily_close.std(axis=0)

    def pull_market_data(self, ticker):
        df = pd.DataFrame()
        try:
            #print(type(ticker), ticker)
            md = yf.Ticker(ticker)
            df = md.history(period='max', start=self.start, end=self.end)
            return df
        except Exception as e:
            print('ERROR @ self.pull_market_data: ',str(e))

    def initialize_charts(self, df):
        ''' Takes the benchmark df and initializes a dataframe for each element
            of a daily candle: Open, High, Low, Close, and Volume'''
        self.daily_open = df['Open'].rename(columns={"Open": self.benchmark}).rename(self.benchmark)
        self.daily_high = df['High'].rename(columns={"High": self.benchmark}).rename(self.benchmark)
        self.daily_low = df['Low'].rename(columns={"Low": self.benchmark}).rename(self.benchmark)
        self.daily_close = df['Close'].rename(columns={"Close": self.benchmark}).rename(self.benchmark)
        self.daily_volume = df['Volume'].rename(columns={"Volume": self.benchmark}).rename(self.benchmark)
        return 0

    def initialize_indicators(self, df):
        ''' Initializes a dataframe for each indicator based on the data in the
            daily candles for the benchmark'''
        return 0

    def add_securities(self, securities):
        ''' Takes a list, securities, and adds each element in the list to our portfolio'''
        securities = [ s for s in securities if s not in self.portfolio]
        securities = [ s for s in securities if self.validate_ticker(s)]
        self.portfolio.extend( securities )

        for asset in self.portfolio:
            df = self.pull_market_data(asset)
            #TODO: modularize code below with self.initialize_charts()
            tmp_open = df['Open'].rename(columns={"Open": asset}).rename(asset)
            tmp_high = df['High'].rename(columns={"High": asset}).rename(asset)
            tmp_low = df['Low'].rename(columns={"Low": asset}).rename(asset)
            tmp_close = df['Close'].rename(columns={"Close": asset}).rename(asset)
            tmp_volume = df['Volume'].rename(columns={"Volume": asset}).rename(asset)

            #print(self.daily_open.name, self.daily_open[:1], '\n\n', tmp_open.name,tmp_open[:1])
            self.daily_open = pd.merge(self.daily_open, tmp_open, on='Date', how='left').bfill()
            self.daily_high = pd.merge(self.daily_high, tmp_high, on='Date', how='left').bfill()
            self.daily_low = pd.merge(self.daily_low, tmp_low, on='Date', how='left').bfill()
            self.daily_close = pd.merge(self.daily_close, tmp_close, on='Date', how='left').bfill()
            self.daily_volume = pd.merge(self.daily_volume, tmp_volume, on='Date', how='left').bfill()



    def remove_securities(self, securities):
        ''' Takes a list, securities, and removes each element from our portfolio'''
        self.portfolio = [ s for s in self.portfolio if s not in securities]

    def calculate_sharpe(self, weights=None):
        expected_returns = self.daily_close.iloc[-1]/self.daily_close.iloc[0]-1
        K = math.sqrt(252)

        contributions = []
        volatilities = []
        for k,v in expected_returns.iteritems():

            asset = {'returns': 0.0, 'numerator': 0.0,
                   'denominator': 0.0, 'sharpe': 0.0,
                   'alloc': 0.0}
            tmp_asset = {}
            # todo: replace SPY with 3 year treasury
            if k != 'SPY': # if we aren't the benchmark
                asset['returns'] = expected_returns[k]
                asset['numerator'] = asset['returns']-expected_returns['SPY']
                asset['denominator'] = p.stdev[k]
                asset['sharpe'] = K*(asset['numerator']/asset['denominator'])
                #todo: allocations based on weights
                if weights == None:
                    asset['alloc'] = 1.0/expected_returns.shape[0]
                else:
                    asset['alloc'] = weights[k]
                tmp_asset[k] = asset
                self.breakdown.append(tmp_asset)
                contributions.append(asset["returns"]*asset["alloc"])
                volatilities.append(asset["denominator"]*asset["alloc"])

        # calculate sharpe of portfolio as a whole, given some array of default allocations and objective
        self.sharpe['returns'] = np.mean(contributions)
        self.sharpe['volatility'] = np.mean(volatilities)
        self.sharpe['sharpe'] = (np.mean(contributions)-expected_returns["SPY"])/np.mean(volatilities)
        return 0

    # TODO: verify that ticker legit
    def validate_ticker(self, ticker):
        return True
#def main():
















# EOF
