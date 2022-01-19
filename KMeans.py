import numpy as np

def sim(a, b):
  # euclidean distance
  return np.linalg.norm(a-b)

def most_sim(ms, x):
  return sorted([(sim(m, x), k) for k,m in ms.items()])[0][1]

class SimpleKMeans:
  n_clusters = None
  n_init = None
  max_iter = None
  tol = None
  means = None
  y = None

  def __init__(self, n_clusters = 3, tol = 0.0001):
    self.n_clusters = n_clusters
    self.tol = tol
  
  def fit_predict(self, X):
    # initial centroids
    cents = np.random.default_rng().choice(X, size=self.n_clusters, replace=False)
    self.means = {
      lab: cents[lab]
      for lab in range(self.n_clusters)
    }

    while True:
      # new cluster labels
      y = np.array([most_sim(self.means, x) for x in X])

      # new cluster means
      means = {
        lab: np.array([
          np.mean(X[y==lab][:, k].flatten()) 
          for k in range(X.shape[1])
        ])
        for lab in range(self.n_clusters)
      }

      # convergence
      if np.all((
        np.absolute(
          np.array(list(self.means.values())) - np.array(list(means.values()))
        )) < self.tol):
        break

      # prepare for next iteration
      self.means = means
      self.y = y
    
    return self.y


if __name__ == "__main__":
  from sklearn.datasets import load_iris
  print(SimpleKMeans(n_clusters=3).fit_predict(load_iris().data))