import numpy as np
import pandas as pd
import os, glob
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')


def most_viewed(df):
    most_viewed = df[["stream_id","times_viewed"]].groupby(["stream_id"]).sum()
    most_viewed = most_viewed.sort_values(by='times_viewed',ascending=False)
    return most_viewed

def to_date(df):
    df['date'] = df['year'].astype(str)+'-'+df['month'].astype(str)+'-'+df['day'].astype(str)
    df['date'] = df['date'].astype('datetime64[ns]')
    df['times_viewed'] = df['times_viewed'].astype('float32')
    df = df.drop(['year','month','day'],axis=1)
    return df

def top_revenue(df):
    most_luc = df[["stream_id","price"]].groupby(["stream_id"]).sum()
    most_luc = most_luc.sort_values(by='price',ascending=False)
    return most_luc

def country_n_stream(df):
    content_streamed = df[['country','stream_id']].groupby(['country']).nunique().rename(columns={'stream_id':'No of contents'})
    content_streamed.drop('country',axis=1,inplace=True)
    return content_streamed.sort_values(ascending=False)

def monthly_revenue_views(df):
    df_monthly = pd.pivot_table(df,values=['price'],index=[pd.Grouper(key='date', freq='M')],columns=['country'],
               aggfunc={'price':np.sum})
    return df_monthly

def total_by_country(df):
    #total views and revenue by country
    by_country = df[['country','price','times_viewed']].groupby(by='country').sum()
    by_country = by_country.sort_values('price',ascending=False).rename(columns={'price':'revenue'})

    others = pd.DataFrame([by_country.sum()-by_country.loc['United Kingdom']],index=['other countries'])
    uk_vs_others = pd.concat([by_country.head(1),others],ignore_index=False)
    print('Top earning countries')
    print('-------------------------')
    print(by_country.head())
    print('')
    print('Least earning countries')
    print('-------------------------')
    print(by_country.tail())
    print('')
    print('UK vs other countries')
    print('-------------------------')
    print(uk_vs_others)

    fig = plt.figure(figsize=(12,5), dpi=80)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_title('Total Global Revenue')
    ax2.set_title('Total Global Revenue')
    by_country.head().plot(kind='bar',ax=ax1,)
    uk_vs_others['revenue'].plot(kind='pie',autopct='%2.1f%%',startangle=0,colormap='cool',ax=ax2)
    plt.show()

def global_monthly_revenue(df):
    # worldwide monthly revenue
    ww_monthly = df[['date','price','times_viewed']].groupby(pd.Grouper(key='date',freq='M')).sum()
    ww_monthly = ww_monthly.rename(columns={'price':'revenue'})
    print(ww_monthly.head())
    ww_monthly.plot(colormap='gnuplot')
    plt.title('Worldwide Monthly Revenue and Streams')
    plt.show()
    return ww_monthly
    
def global_weekly_revenue(df):
    # worldwide weekly revenue
    ww_weekly = df[['date','price','times_viewed']].groupby(pd.Grouper(key='date',freq='W')).sum()
    ww_weekly = ww_weekly.rename(columns={'price':'revenue'})
    print(ww_weekly.head())
    ww_weekly.plot(colormap='gnuplot')
    plt.title('Worldwide Weekly Revenue and Streams')
    plt.show()
    
def global_daily_revenue(df):
    # worldwide daily revenue
    ww_daily = df[['date','price','times_viewed']].groupby(pd.Grouper(key='date',freq='D')).sum()
    ww_daily = ww_daily.rename(columns={'price':'revenue'})
    print(ww_daily.head())
    ww_daily.plot()
    plt.title('Worldwide Daily Revenue and Streams')
    plt.show()