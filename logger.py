

import time, os, re, csv, sys, uuid, joblib
from datetime import date

#if not os.path.exists(os.path.join(".", "logs")):
#    os.mkdir("logs")

def update_train_log(entry_country, entry_date, runtime, MODEL_VERSION, test=False):
    """
    update train log file
    """

    ## name the logfile using something that cycles with date (day, month, year)    
    today = date.today()
    if test:
        logfile = os.path.join("logs", "train-test.log")
    else:
        logfile = os.path.join("logs", "train-{}-{}.log".format(today.year, today.month))

    ## YOUR CODE HERE 
    ## Following the example provided in the Hands on activity of the unit Feedback "loops and unit testing" 
    ## of this course, complete this function to update the train log file at the end of every training.
    ## write the data to a csv file
    header = ['unique_id','timestamp','country','entry_date','runtime','model_version']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.asctime(),entry_country, entry_date, runtime, MODEL_VERSION])
        writer.writerow(to_write)



def update_predict_log(forecast, entry_country, entry_date, std_error_pred, MODEL_VERSION, runtime, test=False):
    """
    update predict log file
    """

    today = date.today()
    if test:
        logfile = os.path.join("logs", "predict-test.log")
    else:
        logfile = os.path.join("logs", "predict-{}-{}.log".format(today.year, today.month))

    ## YOUR CODE HERE 
    ## Following the example provided in the Hands on activity of the unit Feedback "loops and unit testing" 
    ## of this course, complete this function to update the predict log file at the end of every prediction.
    header = ['unique_id','timestamp','forecast','country','entry_date','prediction_standard_error','model_version','runtime']
    write_header = False
    if not os.path.exists(logfile):
        write_header = True
    with open(logfile,'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        if write_header:
            writer.writerow(header)

        to_write = map(str,[uuid.uuid4(),time.asctime(),forecast, entry_country, entry_date, std_error_pred, MODEL_VERSION, runtime])
        writer.writerow(to_write)
        
        
 