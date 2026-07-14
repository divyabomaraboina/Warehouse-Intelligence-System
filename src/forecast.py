import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import load_gold
from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_and_prepare():
    # load gold file
    df = load_gold()
    
    # group by DATE → one row per month
    monthly = df.groupby('DATE')['WAREHOUSE SALES'].sum().reset_index()
    
    # rename → Prophet format
    monthly = monthly.rename(columns={
        'DATE': 'ds',
        'WAREHOUSE SALES': 'y'
    })
    
    print(f"Monthly data shape: {monthly.shape}")
    print(monthly.head())
    return monthly
def train_prophet(df):
    # create Prophet model
    model = Prophet(
        yearly_seasonality = True ,
        weekly_seasonality = False ,
        daily_seasonality = False
    )
    # fit on prepared data
    model.fit(df)
    # return model
    return model

def forecast_future(model , periods =6):
    # create future 6 months dataframe
    #make future dataframe creates past dates + future dates
    # Ms = month start frequency - every month start date 
     future =  model.make_future_dataframe(
         periods= periods , 
         freq = 'MS') 
     #model.predict() -> future dataframe with predictions for each date
     #past dates will have yhat values for actuals and future dates will have yhat values for forecast
     forecast = model.predict(future)
    # ds         → date
    # yhat       → most likely prediction (Bayesian mean)
    # yhat_lower → pessimistic scenario (10% chance below this)
    # yhat_upper → optimistic scenario (10% chance above this)
     print("\n=== NEXT 6 MONTHS FORECAST ===")
    
    # .tail(periods) → last 6 rows only
    # past rows skip chestundi → future predictions matrame chupistundi
     print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))
    # predict on future dates
    # return forecast
     return forecast
def plot_forecast(model, forecast, df_actual):
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # actual data — black dots — original monthly data use cheyyi
    ax.scatter(df_actual['ds'], df_actual['y'], 
               color='black', label='Actual', zorder=5)
    
    # predicted line
    ax.plot(forecast['ds'], forecast['yhat'], 
            color='blue', label='Forecast')
    
    # confidence interval
    ax.fill_between(forecast['ds'], 
                    forecast['yhat_lower'], 
                    forecast['yhat_upper'], 
                    alpha=0.3, color='blue', label='Confidence')
    
    # future 6 months lo values show cheyyi
    for _, row in forecast.tail(6).iterrows():
        ax.annotate(f"${row['yhat']:,.0f}", 
                    xy=(row['ds'], row['yhat']),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center',
                    fontsize=8)
    
    ax.set_title('Warehouse Sales Forecast — Next 6 Months')
    ax.set_xlabel('Date')
    ax.set_ylabel('Warehouse Sales')
    ax.legend()
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'forecast.png')
    plt.show()

def generate_recommendation(forecast):
    
    future_forecast = forecast.tail(6).copy()
    
    print("\n=== NEXT 6 MONTHS FORECAST ===")
    print(f"{'Month':<20} {'Expected':>12} {'Low':>12} {'High':>12} {'Confidence':>12}")
    print("-" * 70)
    
    for _, row in future_forecast.iterrows():
        confidence = (1 - (row['yhat_upper'] - row['yhat_lower']) / row['yhat']) * 100
        print(f"{row['ds'].strftime('%B %Y'):<20} "
              f"${row['yhat']:>11,.0f} "
              f"${row['yhat_lower']:>11,.0f} "
              f"${row['yhat_upper']:>11,.0f} "
              f"{confidence:>11.1f}%")
    
    print("-" * 70)
    
     # Statistical summary
    print(f"\n=== STATISTICAL SUMMARY ===")
    print(f"Average monthly sales:  ${future_forecast['yhat'].mean():,.0f}")
    print(f"Highest month:          ${future_forecast['yhat'].max():,.0f}")
    print(f"Lowest month:           ${future_forecast['yhat'].min():,.0f}")
    print(f"Std deviation:          ${future_forecast['yhat'].std():,.0f}")
    print(f"Total 6 month forecast: ${future_forecast['yhat'].sum():,.0f}")
    
    # Peak and low
    peak = future_forecast.loc[future_forecast['yhat'].idxmax()]
    low  = future_forecast.loc[future_forecast['yhat'].idxmin()]
    
    print(f"\n📈 Peak: {peak['ds'].strftime('%B %Y')} — ${peak['yhat']:,.0f}")
    print(f"   Order by: {(peak['ds'] - pd.DateOffset(months=2)).strftime('%B %Y')}")
    print(f"\n📉 Lowest: {low['ds'].strftime('%B %Y')} — ${low['yhat']:,.0f}")
    print(f"   Reduce orders in: {(low['ds'] - pd.DateOffset(months=1)).strftime('%B %Y')}")
    print("=" * 60)
if __name__== "__main__":
    df = load_and_prepare()
    model = train_prophet(df)
    print("Prophet model trained successfully.")
    forecast = forecast_future(model)
    plot_forecast(model, forecast,df)
    generate_recommendation(forecast)