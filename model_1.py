import numpy as np
import pandas as pd
import os, glob, csv, joblib, re
from statsmodels.tsa.arima_model import ARIMA
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')

train_dir = os.path.join('ai-workflow-capstone','cs-train')
production_dir = os.path.join('ai-workflow-capstone','cs-production')
json_files = glob.glob(os.path.join(train_dir,'invoices**.JSON'))  
filepath = os.path.join('data','processed_data.csv')

MODEL_VERSION = 1.1
SAVED_MODEL = os.path.join("saved_models", "revenue-forecast-{}.joblib".format(re.sub("\.","_", str(MODEL_VERSION))))


class load_data():
    def __init__(self,
                 files_description, #example: 'invoices**.JSON'
                ):
        self.desc = files_description
        
    def fit(self,data_dir,y=None):
        files = glob.glob(os.path.join(data_dir,self.desc))
        df = pd.read_json(files[0])
        self.col = list(df.columns)
        return self
        
    def transform(self,data_dir):
        files = glob.glob(os.path.join(data_dir,self.desc))
        df = pd.DataFrame([],columns=self.col)
        for i in range(len(files)):
            df_join = pd.read_json(files[i])
            df_join.columns=self.col
            df = pd.concat([df,df_join],ignore_index=True)
        return df
    
class preprocess_data():
    def __init__(self,saved_path,remove_existing=False):
        self.reset = remove_existing
        self.path = saved_path
        
    def extract(self,df):
        #remove negative price
        df.drop(index=df.index[(df.price<0)&(df.price>2000)],inplace=True)
        
        df['date'] = df['year'].astype(str)+'-'+df['month'].astype(str)+'-'+df['day'].astype(str)
        df['date'] = df['date'].astype('datetime64[ns]')
        df1 = df.drop(['year','month','day'],axis=1)
        return df1[self.fields]
        
    def write_csv(self,df):
        write_header = False
        if not os.path.exists(self.path):
            write_header = True
        with open(self.path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            if write_header:
                writer.writerow(self.fields)
            writer.writerows(df.reset_index().to_numpy())
                    
    def fit(self,df=None,y=None):
        if self.reset:
            if os.path.exists(self.path):
                os.remove(self.path)
        self.fields = ['date','country','price','times_viewed']
        return self

    def transform(self,df):
        df_extract = self.extract(df)
        df_processed = pd.pivot_table(df_extract,values=['price','times_viewed'],index=[pd.Grouper(key='date', freq='D'),'country'],
                                        aggfunc={'price':np.sum, 'times_viewed':np.sum})
        self.write_csv(df_processed)
        return df_processed
    
class load_extract():
    def country_wise(self,df,country,date):
        df = df.copy()
        new_Index = pd.MultiIndex.from_product([self.date_range,self.country_array], names=['date', 'country'])
        df.set_index(['date','country'],inplace=True)   
        df = df.reindex(new_Index)
        df.fillna(0,inplace=True)
        if date==None:
            date = df.reset_index().date.iloc[-1]
        sample_df = df.xs(country, level=1).loc[:date]
        return sample_df
    
    def all_countries(self,df,date):
        df = df.copy()
        df = df.groupby('date').sum()
        new_Index = pd.Index(self.date_range,name='date')
        df = df.reindex(new_Index)
        df.fillna(0,inplace=True)
        if date==None:
            date = df.reset_index().date.iloc[-1]
        sample_df = df.loc[:date]
        return sample_df
        
    def fit(self,X=None,y=None):
        df = pd.read_csv(filepath)
        df.drop_duplicates(inplace=True)
        df['date'] = df['date'].astype('datetime64[ns]')
        self.date_array = df.date.unique()
        self.country_array = df.country.unique()
        self.date_range = pd.date_range(self.date_array[0],self.date_array[-1],freq='D')
        self.country_list = df.country.values
        self.df = df
        return self
    
    def transform(self,X,y=None):
        """X is a json {country: date:}"""
        country = X['country']
        if 'date' not in X:
            date = None
        elif X['date'] in self.date_range:
            date = X['date']
        else:
            raise NameError("""invalid date: select a date between {} and {} 
                            or remove 'date' from query""".format(self.date_range[0],self.date_range[-1]))
        if country.lower() in ['all','all countries']:
            daily_df = self.all_countries(self.df,date) 
            
        elif country in self.country_list:
            daily_df = self.country_wise(self.df,country,date)
        else:
            raise NameError('invalid country name: available country names are - {}'.format(self.country_list))
        return daily_df
    
class XGR_forecast():
    def __init__(self,period=30,n_steps=5,n_estimators=200,max_depth=2, min_child_weight=100, colsample_bytree=0.4,
                        tree_method='exact', eta=0.33, seed=42, col='price',
                         gamma=1.0,reg_alpha=1.0,reg_lambda=1.0):
        self.period=period
        self.n_steps = n_steps
        self.n = n_estimators
        self.depth = max_depth
        self.child = min_child_weight
        self.cbt = colsample_bytree
        self.tm = tree_method
        self.eta = eta
        self.seed = seed
        self.col = col
        self.gm = gamma
        self.ra = reg_alpha
        self.rl = reg_lambda
    
    def create_features(self, x_data, y_data, n_steps):
        """ Use sliding window approach to generate features"""
        X, y = list(), list()
        n_samples = x_data.shape[0]
        for i in range(n_samples ):
            # compute a new (sliding window) index
            end_ix = i + n_steps
            # if index is larger than the size of the dataset, we stop
            if end_ix < n_samples:
                # Get a sequence of data for x
                seq_X = x_data[i:end_ix]
                # Get only the last element of the sequency for y
                seq_y = y_data[end_ix]
                # Append the list with sequencies
                X.append(seq_X)
                y.append(seq_y)

            elif end_ix == n_samples:
                x_test = x_data[i:end_ix].flatten()
            else:
                break
        # Make final arrays
        x_array = np.array(X)
        y_array = np.array(y)
        #reshape x_array to 2d array if x_data is 2d
        try:  
            l,_,w = x_array.shape
            x_array = x_array.reshape(l,w*n_steps)
        except:
            pass
        return x_array,x_test,y_array
    
    def XGR_itr_forecast(self,series,n_steps):
        frcst = []
        for i in range(self.period):
            X,X_test,y = self.create_features(series, series, n_steps)
            X_test = X_test.reshape(-1,n_steps)
            model = self.XGR_trainer(X,y)
            y_new = model.predict(X_test)
            series = np.hstack((series,y_new))
            frcst.append(y_new[0])
        return np.array(frcst)
    
    def XGR_trainer(self,X_train,y_train):
        xgr = XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.5,
                       min_child_weight=200,
                      colsample_bytree=0.3,
                       tree_method='exact',  #this makes training slower but more accurate
                        eta=0.13,    
                        seed=42)
        self.model = xgr.fit(X_train,y_train,
                    eval_metric="mae", 
                    eval_set=[(X_train,y_train)], 
                    verbose=0, 
                    #early_stopping_rounds = 10,
                   )
        return self.model
    
    def fit(self,df,y=None):
        series = df[self.col].to_numpy()
        self.frcst = self.XGR_itr_forecast(series,self.n_steps)
        return self
    
    def predict(self,y=None):
        return self.frcst
    
    def eval_result(self):
        return self.model.evals_result_['validation_0']['mae'][-1]
    
class ARIMA_forecast():
    def __init__(self,period=30,p=1,n_steps=5,d=0,typ='levels',dynamic=True,col='price'):
        self.period = period
        self.n_steps = n_steps
        self.d = d
        self.p = p
        self.typ = typ
        self.dynamic = dynamic
        self.col = col
        
    def fit(self,df,y=None):
        series = df[self.col].to_numpy()
        self.l = len(series)
        m = ARIMA(series, order=(self.p,self.d,self.n_steps))
        try:
            self.m_fit = m.fit()
        except:
            m = ARIMA(series, order=(self.p,1,3))
            self.m_fit = m.fit()
        return self
    
    def predict(self,df=None):
        forecast = self.m_fit.predict(start=self.l,end=self.l+self.period,typ=self.typ,dynamic=self.dynamic)
        return forecast
    def eval_result(self):
        return self.m_fit.bse[0]

class AVG_forecast:
    def __init__(self,n_steps=5, period=30):
        self.n_steps = n_steps
        self.period = period
        
    def avg_itr(self,series):
        forecast = []
        for i in range(self.period):
            new = np.array(sum(series)/self.n_steps)
            series = np.hstack((series[1:],new))
            forecast.append(new)
        return np.array(forecast)
        
    def fit(self,df,y=None):
        self.series = df.tail(self.n_steps).price.to_numpy()        
        return self
    
    def predict(self,df=None):
        return self.avg_itr(self.series)
    
def model_iter(models,all_df):
    l = len(all_df)
    idx = [l-120,l-90,l-60,l-30]
    df_test0 = all_df.iloc[idx[0]:idx[1]]
    df_test1 = all_df.iloc[idx[1]:idx[2]]
    df_test2 = all_df.iloc[idx[2]:idx[3]]
    df_test3 = all_df.iloc[idx[3]:]
    y_true = [dfm.sum().price for dfm in [df_test0,df_test1,df_test2,df_test3]]
    Y_pred = []
    mae = []
    for mdl in models:
        y_pred = []
        for i in range(1,len(idx)):
            forecast = mdl.fit(all_df.iloc[:idx[i]]).predict()
            y_pred.append(sum(forecast))
        Y_pred.append([y_true[0]]+y_pred)
        mae.append(mean_absolute_error(y_true[1:],y_pred))
        
    trn = all_df.iloc[:idx[1]]
    offset = len(trn)%30
    train_set = trn.iloc[offset:].reset_index().groupby(pd.Grouper(key='date',freq='30D')).sum()
        
    Y_pred.append(y_true)
    strt = train_set.reset_index().date.iloc[-1]
    test_index = pd.date_range(start=strt,periods=len(idx),freq='30D')

    cols = ['Averaging','XGR','ARIMA','True_revenue']
    df_pred = pd.DataFrame((np.array(Y_pred)).T, columns=cols,index=test_index)
    train_set = train_set.rename(columns={'price':'training_set'})
    
    fig = plt.figure(figsize=(15,5),dpi=80)
    ax = fig.add_subplot()
    train_set.plot(y='training_set',ax=ax)
    df_pred.plot(ax=ax)
    
    df_mae = pd.DataFrame(mae,index=cols[:3],columns=['mean_absolute_error'])
    print(df_mae)
        
def model_train(query):
    load = load_data('invoices**.JSON')
    prep = preprocess_data(filepath)
    prep_pipe = Pipeline(steps=[("load",load),('prepare',prep),])
    frcst_pipe = Pipeline(steps=[("lde",load_extract()),
                       ('forecast',XGR_forecast()),])
    if not os.path.exists(filepath):
        prep_pipe.fit(train_dir).transform(train_dir)
    model = frcst_pipe.fit(query)
    joblib.dump(model,SAVED_MODEL)
    return model

def model_load():
    if not os.path.exists(SAVED_MODEL):
        raise Exception("Model '{}' cannot be found did you train the model?".format(SAVED_MODEL))
    return joblib.load(SAVED_MODEL)

def model_predict(query,train=True):
    if train:
        model = model_train(query)
    else:
        model = model_load()
    forecast = model.predict(query)
    return forecast

def model_eval():
    model = model_load()
    return model.named_steps.forecast.eval_result()

#make a forecast  
query = {'country':'Belgium','date':'2019-06-30'}
forecast = model_predict(query)
if 'date' not in query:
    print('The projected revenue for next month is {}'.format(int(sum(forecast))))
else:
    start_date = query['date']
    end_date = pd.date_range(start=start_date,periods=2,freq='30D').astype(str)[1]
    print('The projected revenue from {} to {} is {}'.format(start_date,end_date,int(sum(forecast))))














