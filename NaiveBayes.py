import numpy as np

class SimpleNaiveBayes:
  root = None
  
  def __init__(self):
    pass

  def fit(self, X, y):
    return self

  def predict(self, X):
    pass

if __name__ == "__main__":
  from sklearn.datasets import load_iris
  iris = load_iris()
  clf = SimpleNaiveBayes().fit(iris.data, iris.target)
  print(clf.predict(iris.data))