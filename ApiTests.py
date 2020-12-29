#!/usr/bin/env python
"""
api tests

these tests use the requests package however similar requests can be made with curl

e.g.

data = '{"key":"value"}'
curl -X POST -H "Content-Type: application/json" -d "%s" http://localhost:8080/predict'%(data)
"""

import sys, json
import os
import unittest
import requests
import re
from ast import literal_eval
import numpy as np

port = 8882

try:
    requests.post('http://127.0.0.1:{}/predict'.format(port))
    server_available = True
except:
    server_available = False
    
## test class for the main window function
class ApiTest(unittest.TestCase):
    """
    test the essential functionality
    """

    @unittest.skipUnless(server_available, "local server is not running")
    def test_01_train(self):
        """
        test the train functionality
        """
      
        query = {'country':'Belgium','date':'2019-06-30'}
        r = requests.post('http://127.0.0.1:{}/train'.format(port), json=query)
        train_complete = re.sub("\W+","",r.text)
        self.assertEqual(train_complete,'true')
    
    @unittest.skipUnless(server_available, "local server is not running")
    def test_02_predict(self):
        """
        test the predict functionality
        """

        query = {'country':'Belgium','date':'2019-06-30'}
        r = requests.post('http://127.0.0.1:{}/predict'.format(port), json=query)
        response = literal_eval(r.text)
        print(response)
        self.assertTrue((response['forecast']>0)&(response['forecast']<50000))


    @unittest.skipUnless(server_available,"local server is not running")
    def test_04_logs(self):
        """
        test the log functionality
        """

        file_name = 'train-test.log'
        request_json = {'file':'train-test.log'}
        r = requests.get('http://127.0.0.1:{}/logs/{}'.format(port, file_name))

        with open(file_name, 'wb') as f:
            f.write(r.content)
        
        self.assertTrue(os.path.exists(file_name))

        if os.path.exists(file_name):
            os.remove(file_name)

        
### Run the tests
if __name__ == '__main__':
    unittest.main()
