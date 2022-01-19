class SimpleDBSCAN:
  eps = None

  def __init__(self, n_clusters = None):
    self.eps = eps
  
  def fit_predict(self, X):
    pass

if __name__ == "__main__":
  from sklearn.datasets import load_iris
  print(SimpleDBSCAN()(eps=0.2).fit_predict(load_iris().data))