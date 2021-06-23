import pandas as pd
import numpy as np
from datetime import date
import time
from alpha_vantage.timeseries import TimeSeries
 

# get tickers of assets that were part of ^IBOV at least once from 2010-2021
def get_tickers():
    ibov_tickers = set([]) # initialize an empty set of tickers
    filepath = 'data/IBOV_comp_data/'
    quads = [1,2,3] # quad index
    years = [i for i in range(2010,2022)] # year index

    for y in years:
        for q in quads: 
            # check if current quad data exists
            filename = filepath + 'quad' + str(q) + '_' + str(y) + '.xlsx'
            try:
                IBOV_comp = pd.read_excel(filename)
                IBOV_comp = IBOV_comp.dropna(axis = 0)
                #print(IBOV_comp)
                curr_tickers_set = set(IBOV_comp["COD."].values.tolist()) # get tickers for current year and quad
                ibov_tickers = ibov_tickers.union(curr_tickers_set)

            except FileNotFoundError:
                print(filename + ' NOT found...')
    #print(list(ibov_tickers))
    #print(len(list(ibov_tickers)))
    return list(ibov_tickers)


# --------------------------------------------- ----------- ----------- --------------------------------------------- #
# --------------------------------------------- MAIN SCRIPT BEGINS HERE --------------------------------------------- #
# --------------------------------------------- ----------- ----------- --------------------------------------------- #
# initialize the full data set 
final_DB = dict([])
#alpha_vantage requests per day: 500 / requests per minute: 5
# alpha vantage doc: https://alpha-vantage.readthedocs.io/en/latest/
alpha_key='KGJAQOF0FSLFOH8P'
ts = TimeSeries(key = alpha_key, output_format = 'pandas')

# search parameters
all_tickers = ["IBOV"] # index added...
all_tickers.extend([ticker for ticker in get_tickers()])
print("total number of requests: " + str(len(all_tickers)))
# filter parameters
start_date ='2010-01-12' # yyyy-mm-dd
end_date ='2021-05-07' # yyyy-mm-dd
tol_missing_values = 5 # stock's missing values tolerance

idx = 0
n_requests = 0 # used to trigger sleep()
max_days = -1 # get the number of days in the stock data
while idx in range(len(all_tickers)): # loop over all tickers

    filepath = 'data/source_alphavantage/' + all_tickers[idx] + '.csv'
    # check if this ticker data was already gathered
    try:
        df = pd.read_csv(filepath)
        print(all_tickers[idx] + ' data found')
        # first lets ensure that the date column is 
        df['date'] = pd.to_datetime(df['date']) 
        # next, set the mask -- we can then apply this to the df to filter it
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        # assign mask to df to return the rows with date between our specified start/end dates
        #print(df)
        df = df.loc[mask]

        # get IBOV dates
        if all_tickers[idx] == 'IBOV':
            final_DB['date'] = list(df['date'])
            ibov_dates = final_DB['date']

        # create a dict to associate a price to a date to incrementaly synchronize with IBOV dates
        hist_close = dict([])
        histData = list(df['4. close'])
        histDates = list(df['date'])
        for idx_price in range(len(histDates)):
            hist_close[histDates[idx_price]] = histData[idx_price] 

        # adjust/synchronize hist close prices of this stock with IBOV close prices. If there is missing data, insert None in that date
        adjusted_histData = []
        for date in final_DB['date']: # loop over IBOV dates
            if date in hist_close.keys():
                adjusted_histData.append(hist_close[date])
            else:
                adjusted_histData.append(None)

        final_DB[all_tickers[idx]] = adjusted_histData  
        idx += 1 # go to the next ticker.... 
        
    # make request to the alpha_vantage API to gather ticker's data
    except FileNotFoundError:
        #search = ts.get_symbol_search(keywords = all_tickers[idx])
        if n_requests == 5:
            n_requests = 0
            print("5 requests limit per minute reached")
            print("sleeping for 2 min before starting to send requests again...")
            time.sleep(120)  # Delay for 2 min (120 seconds).
        print(all_tickers[idx] + ' data not found. Collecting data using the alpha vantage API....')
        try:
            df, meta_data = ts.get_daily(symbol = all_tickers[idx] + '.SAO', outputsize = 'full')
            # get the individual stock data and then save to .csv
            df.to_csv(filepath)
            print("done!")
        except ValueError:
            print(all_tickers[idx] + " data was not found by the alpha vantage API...")
            idx += 1 # go to the next ticker.... 
        n_requests += 1
        print("current alphavantage requests: " + str(n_requests))

#df = pd.DataFrame.from_dict(final_DB, orient='index')
#df = df.transpose()
df = pd.DataFrame.from_dict(final_DB)
print("total number of assets contained in IBOV_DB raw: " + str(len(df.columns) - 1))
print(df)

# remove tickers with more missing values than we can tolerate
final_tickers = list(df.columns)
for ticker in final_tickers[1:]:
    #print(ticker + " NaN count: " + str(df[ticker].isna().sum()))
    remove = False
    if df[ticker].isna().sum() > tol_missing_values :
        df = df.drop(labels = ticker, axis = 1)

df = df.dropna(axis = 0)
print("total number of assets contained in IBOV_DB.csv: " + str(len(df.columns) - 1))
print(df)

df.to_excel('data/IBOV_DB.xlsx')
print('data/IBOV_DB.xlsx was written')


'''
# plot if desired...
data['close'].plot()
plt.title('Daily Times Series for the MSFT stock')
plt.show()
'''