class SimpleKMeans:
  n_clusters = None

  def __init__(self, n_clusters = None):
    self.n_clusters = n_clusters
  
  def fit_predict(self, X):
    pass

if __name__ == "__main__":
  from sklearn.datasets import load_iris
  print(SimpleKMeans(n_clusters=3).fit_predict(load_iris().data))