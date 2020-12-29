#!/usr/bin/env python
"""
model tests
"""

import os, sys
import csv
import unittest
from ast import literal_eval
import pandas as pd
import numpy as np
from datetime import date
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
from logger import update_train_log, update_predict_log


class LoggerTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        ensure log file is created
        """

        log_file = os.path.join("logs", "train-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        
        ## YOUR CODE HERE
        ## Call the update_train_log() function from logger.py with arbitrary input values and test if the log file 
        ## exists in you file system using the assertTrue() base method from unittest.
        inpt = ['Nigeria',100,0.95,2.55]
        update_train_log(*inpt,test=True)
        self.assertTrue(os.path.exists(log_file))

        
    def test_02_train(self):
        """
        ensure that content can be retrieved from log file
        """
        
        log_file = os.path.join("logs", "train-test.log")
    
        ## YOUR CODE HERE
        ## Log arbitrary values calling update_train_log from logger.py. Then load the data
        ## from this log file and assert that the loaded data is the same as the data you logged.
        inpt = ['Nigeria',100,0.95,2.55]
        log = pd.read_csv(log_file)
        outpt = list(log.values[0][2:])
        self.assertEqual(inpt,outpt)


    def test_03_predict(self):
        """
        ensure log file is created
        """

        log_file = os.path.join("logs", "predict-test.log")
        if os.path.exists(log_file):
            os.remove(log_file)
        
        ## YOUR CODE HERE
        ## Call the update_predict_log() function from logger.py with arbitrary input values and test if the log file 
        ## exists in you file system using the assertTrue() base method from unittest.
        inpt = ['Nigeria',0.99,100,1.2,0.5,1.0]
        update_predict_log(*inpt, test=True)
        self.assertTrue(os.path.exists(log_file))

    
    def test_04_predict(self):
        """
        ensure that content can be retrieved from log file
        """ 

        log_file = os.path.join("logs", "predict-test.log")
    
        ## YOUR CODE HERE
        ## Log arbitrary values calling update_predict_log from logger.py. Then load the data
        ## from this log file and assert that the loaded data is the same as the data you logged.
        inpt = ['Nigeria',0.99,100,1.2,0.5,1.0]
        log = pd.read_csv(log_file)
        outpt = list(log.values[0][2:])
        self.assertEqual(inpt,outpt)

### Run the tests
if __name__ == '__main__':
    unittest.main()
      
