import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
import sys

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



#def main():
















# EOF
