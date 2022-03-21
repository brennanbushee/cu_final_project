costco.index = pd.to_datetime(costco.index)

#%%

spy = pd.read_csv('data/SPY.csv', index_col='Date', usecols= ['Date', 'Adj Close'])
spy = spy.rename(columns={"Adj Close": 'SPY'})
spy.index = pd.to_datetime(spy.index)
spy_train = spy[:'2020-01-01']
spy_test = spy['2020-01-01':].pct_change()[1:]

#%%

costco.head(5)

#%%

equal_weight_return = combined['2020-01-01':].pct_change()[1:]

#%%

equal_weight_return = equal_weight_return.mean(axis=1)

#%%

symbols = ['MU', 'SNBR', 'GOOGL', 'MIDD', 'CMPR', 'LSTR', 'SSD', 'SNA', 'TROW', 'V', 'CHH', 'HIFS',
           'CRI', 'LANC', 'TTC', 'MNST']
combined = costco
for stock in symbols:
    prices = pd.read_csv('data/' + stock + '.csv', index_col='Date', usecols= ['Date', 'Adj Close'])
    prices = prices.rename(columns={"Adj Close": stock})
    prices.index = pd.to_datetime(prices.index)
    combined = combined.merge(prices, on='Date')
combined.head(5)    

#%%

#Split out your trading set and set your means and variances.
training_set = combined[:'2020-01-01']
mu = pypfopt.expected_returns.mean_historical_return(training_set)
S = pypfopt.risk_models.semicovariance(training_set)


#%%

hrp = pypfopt.HRPOpt(combined.pct_change()[1:],S)

#%%

hrp_weights = hrp.optimize(linkage_method='ward')

#%%

#Visualize clusters of stocks.
from pypfopt import plotting

plotting.plot_dendrogram(hrp);


#%%

#One possible comparison portfolio: the ten stocks with the highest alpha over the training period.
ten_highest_alphas = mu.sort_values(ascending=False)[:10].keys()

#%%

ten_highest_alphas

#%%

combined_highest_alpha = combined[ten_highest_alphas]
highest_alpha_returns = combined_highest_alpha.pct_change()[1:].mean(axis=1)

#%%

#Baseline: traditional min vol and capm returns.
ef = pypfopt.EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
#ef.save_weights_to_file("weights.csv") # saves to file
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

#%%

cleaned_weights

#%%

cleaned_weights

#%%

ef.set_weights(cleaned_weights)
ef.portfolio_performance(verbose=True)

#%%

ef = pypfopt.EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
#ef.save_weights_to_file("weights.csv") # saves to file
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

#%%

cleaned_weights

#%%



#%%

ef.set_weights(equal_weights)
ef.portfolio_performance(verbose=True)

#%%

#Test set here.
test_set = combined['2020-01-01':].pct_change()[1:]
max_sortino_returns = (test_set * cleaned_weights).sum(axis=1)
#kelly_returns = (test_set * kelly_fractions).sum(axis=1)

#%%

ef.set_weights(kelly_fractions)
ef.portfolio_performance(verbose=True)

#%%

ef.set_weights(current_weights)

#%%

spy.pct_change()[1]

#%%

test_set = combined['2020-01-01':].pct_change()[1:]
max_sortino_returns = (test_set * cleaned_weights).sum(axis=1)
#kelly_returns = (test_set * kelly_fractions).sum(axis=1)

#%%

ef.set_weights(kelly_fractions)
ef.portfolio_performance(verbose=True)

#%%

ef.set_weights(current_weights)

#%%

spy.pct_change()[1]

#%%

test_set = combined['2020-01-01':].pct_change()[1:]
#max_sortino_returns = (test_set * cleaned_weights).sum(axis=1)
#kelly_returns = (test_set * kelly_fractions).sum(axis=1)
hrp_weight_returns = (test_set * hrp_weights).sum(axis=1)

#%%

test_set = combined['2020-01-01':].pct_change()[1:]
max_sortino_returns = (test_set * cleaned_weights).sum(axis=1)
#kelly_returns = (test_set * kelly_fractions).sum(axis=1)

#%%

ef.set_weights(kelly_fractions)
ef.portfolio_performance(verbose=True)

#%%

spy.pct_change()[1]

#%%

test_set = combined['2020-01-01':].pct_change()[1:]

hrp_weight_returns = (test_set * hrp_weights).sum(axis=1)

#%%

test_set = combined_highest_alpha['2020-01-01':].pct_change()[1:]
max_sortino_returns = (test_set * cleaned_weights).sum(axis=1)
#kelly_returns = (test_set * kelly_fractions).sum(axis=1)

#%%

max_sortino_returns

#%%

equal_weights = {'COST': 1/17}
for ticker in symbols:
    equal_weights[ticker] = 1/17
equal_weights
    

#%%

kelly_weights = equal_weights.copy()
for ticker in kelly_weights.keys():
    kelly_weights[ticker] = kelly_fraction(annual_returns(ticker))
kelly_weights
    

#%%

daily_returns = combined.pct_change() + 1
monthly_returns = daily_returns.resample('M').prod()



#%%

np.sum(current_weights * combined.pct_change(periods=252).loc['2021-01-04'])

#%%

ef = pypfopt.EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
#ef.save_weights_to_file("weights.csv") # saves to file
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

#%%



#%%

ef.set_weights(equal_weights)
ef.portfolio_performance(verbose=True)

#%%

ef.set_weights(current_weights)
ef.portfolio_performance(verbose=True)

#%%

equal_weights = {'COST': 1/17}
for ticker in symbols:
    equal_weights[ticker] = 1/17
equal_weights
    

#%%

daily_returns = combined.pct_change() + 1
monthly_returns = daily_returns.resample('M').prod()

annual_returns = combined.pct_change(periods=252)['2017-03-13':]

#%%



#%%

portfolio_value = kelly_fractions * combined[2020:]

#%%

ef.portfolio_performance?

#%%

def kelly_fraction(returns):
    returns = np.array(returns)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    W = len(wins) / len(returns)
    R = np.mean(wins) / np.abs(np.mean(losses))
    kelly_f = W - ( (1 - W) / R )
    return kelly_f

#%%

monthly_returns.cumprod()

#%%

quarterly_returns = (daily_returns).resample('3M').prod()
#quarterly_returns.mask(quarterly_returns < -0.1)
#mean_return = quarterly_returns.mean(axis=1)
#mean_return.name = "Portfolio mean"
#quarterly_returns.join(mean_return)
#quarterly_returns[quarterly_returns > -0.1].columns()
quarterly_returns['2020']


#%%

#Copy-pasted from my reading report. One other potential allocation strategy.
def kelly_fraction(returns):
    returns = np.array(returns)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    W = len(wins) / len(returns)
    R = np.mean(wins) / np.abs(np.mean(losses))
    kelly_f = W - ( (1 - W) / R )
    return kelly_f
