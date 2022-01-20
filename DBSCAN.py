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

  def __init__(self, pos, eps_nhood = None, cluster = None, core = False, visited = False):
    self.pos = pos
    self.core = core
    self.eps_nhood = eps_nhood
    self.visited = visited
    self.cluster = cluster
  
  def __sub__(self, other):
    return self.pos - other.pos

class SimpleDBSCAN:
  eps = None
  min_samples = None
  pts = None
  n_clusters = None

  def __init__(self, eps = 0.2, min_samples = 5, pts = None, n_clusters = 0):
    self.eps = eps
    self.min_samples = min_samples
    self.pts = pts
    self.n_clusters = n_clusters
  
  def fit_predict(self, X):
    self.pts = np.array([SimpleDBSCANNode(pos=x) for x in X])
    for p in self.pts:
      p.eps_nhood = self.pts[[dist(p, pp) <= self.eps for pp in self.pts]]
    for p in np.random.default_rng().choice(self.pts, size=len(self.pts), replace=False):
      if not p.visited:
        p.visited = True
        if len(p.eps_nhood) >= self.min_samples:
          p.cluster = self.n_clusters
          self.n_clusters += 1
          n = [pp for pp in p.eps_nhood] # want same elements, but not same array
          for pp in n:
            if not pp.visited:
              pp.visited = True
              if len(pp.eps_nhood) >= self.min_samples:
                n += list(pp.eps_nhood)
            if pp.cluster is None:
              pp.cluster = p.cluster
        else:
          # p.cluster is None when "noise" is denoted
          pass
    
    return np.array([p.cluster for p in self.pts])

if __name__ == "__main__":
  from sklearn.datasets import load_iris
  print(SimpleDBSCAN().fit_predict(load_iris().data))