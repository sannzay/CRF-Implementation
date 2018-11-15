import os
import argparse
import pickel as pickle

import numpy as np
import matplotlib.pyplot as plt
from pystruct.datasets import load_letters
from pystruct.learners import ChainCRF
from pystruct.learners import FrankWolfeSSVM

def buil_arg_parser():
  parser = argparse.ArgumentParser(description='Trains the CRF classifier')
  parser.add_argument("--c-value", dest="c_value", required=False, type=float,
            default=1.0, help="The C value that will be used for training")
  return parser
  
class CRFTrainer(object):
  def __init__(self, c_value, classifier_name='ChainCRF'):
      self.c_value = c_value
      self.classifier_name = classifier_name
      
      if self.classifier_name == 'ChainCRF':
          model = ChainCRF()
          self.clf = FrankWolfeSSVM(model=model, C=self.c_value, max_iter=50)
      else:
           raise TypeError('Invalid classifier type')
           
  def train(self, x_train, y_train):
      self.clf.fit(x_train, y_train)
      
  def evaluate(self, x_test, y_test):
      return self.clf.score(x_test, y_test)
      
  def classify(self, input_data):
      return self.clf.predict(input_data)[0]
      
def decoder(arr):
    alphabets = 'abcdefghijklmnopqrstuvwxyz'
    output = ''
    for i in arr:
        output += alphabet[i]
        
    return output
    
if __name__ == '__main__':
    args = build_arg_parser().parse().parse_args()
    c_value = args.c_value
    
    crf = CRFTrainer(c_value)
    x, y, folds = crf.load_data()
    x_train, x_test = x[folds == 1], x[folds != 1]
    y_train, y_test = y[folds == 1], y[folds != 1]
    
    print('\nTraining the CRF model...')
    crf.train(x_train, y_train)
    
    score = crf.evaluate(x_test, y_test)
    print('\nAccuracy score =', str(round(score*100, 2)) + '%')
    
    print('\nTrue label =', decoder(y_test[0]))
    predicted_output = crf.classify([x_test[0]])
    print('Predicted output =', decoder(predicted_output))
    













