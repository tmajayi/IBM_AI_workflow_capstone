#!/usr/bin/env python
"""
model tests
"""


import unittest
## import model specific functions and variables
from model import *

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        query = {'country':'Belgium','date':'2019-06-30'}
        model_train(query)
        self.assertTrue(os.path.exists(SAVED_MODEL))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## load the model
        model = model_load()
        self.assertTrue('predict' in dir(model))
        self.assertTrue('fit' in dir(model))

    def test_03_predict(self):
        """
        test the predict functionality
        """

        ## load model first
        query = {'country':'Belgium','date':'2019-06-30'}
        model = model_load()
        
        train_dir = os.path.join('ai-workflow-capstone','cs-train')
        forecast = model.predict(query)
        self.assertTrue((forecast>0)&(forecast<10000))
        
        

        
### Run the tests
if __name__ == '__main__':
    unittest.main()
