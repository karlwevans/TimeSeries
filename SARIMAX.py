# Import required modules
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Explore the data
team_flights = pd.read_csv('./team_flights.csv')
fuel_prices = pd.read_csv('./fuel_prices_2101.csv',
                         index_col='date')

# Some basic data cleaning and pre-processing
team_flights['departure_datetime'] = pd.to_datetime(team_flights['departure_datetime'])
team_flights['landing_datetime']   = pd.to_datetime(team_flights['landing_datetime'])

fuel_prices.index = pd.DatetimeIndex(fuel_prices.index).to_period('D')

# 1: Max flights in the air
# Pivot DateTimes
datetimes=pd.concat([team_flights['departure_datetime'],team_flights['landing_datetime']])
datetimes_sorted=sorted(list(set(datetimes)))

# Prep DF
df=pd.DataFrame({'date':sorted(datetimes_sorted), 'in_flight':0})

for index, flight in team_flights.iterrows():
    df.loc[(df['date']>=flight['departure_datetime']) & (df['date']<flight['landing_datetime']), 'in_flight'] +=1

# Optional 
plt.plot('date','in_flight',data=df)   
# Store
max_teams_in_flight=df['in_flight'].max()

# 2 Total Fuel Expense
# Model future fuel costs
model=SARIMAX(fuel_prices, order=(1,1,1),seasonal_order=(1,0,0,7))
model_fit=model.fit()
forecast=model_fit.get_forecast(steps=365)
predicts=pd.DataFrame(data={"date":forecast.summary_frame().index.to_timestamp(),"price":forecast.predicted_mean.values})

# Get flights and dates
team_flights['departure_date']=team_flights['departure_datetime'].dt.date
predicts['date']=predicts['date'].dt.date
predicts.set_index('date',inplace=True)

# Join and multiply
predicts=team_flights.join(predicts, on='departure_date', how='left')
predicts['fuel_costs']=predicts['travel_distance_miles']*predicts['price']
# Store total
total_fuel_spend_2102_dollars=predicts['fuel_costs'].sum()
