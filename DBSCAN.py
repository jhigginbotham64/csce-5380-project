import numpy as np

def dist(a, b):
  # euclidean distance
  return np.linalg.norm(a-b)

class SimpleDBSCANNode:
  visited = None
  core = None
  pos = None
  eps_nhood = None
  cluster = None

  def __init__(self, pos, eps_nhood, cluster = None, core = False, visited = False):
    self.pos = pos
    self.core = core
    self.eps_nhood = eps_nhood
    self.visited = visited
    self.cluster = cluster

class SimpleDBSCAN:
  eps = None
  min_samples = None

  def __init__(self, eps = None, min_samples = 5):
    self.eps = eps
    self.min_samples = min_samples
  
  def fit_predict(self, X):
    pass

if __name__ == "__main__":
  from sklearn.datasets import load_iris
  print(SimpleDBSCAN(eps=0.2).fit_predict(load_iris().data))