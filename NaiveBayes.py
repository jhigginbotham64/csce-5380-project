import numpy as np

class SimpleGaussian:
  u = None
  sig = None

  def __init__(self, u, sig):
    self.u = u
    self.sig = sig
  
  def __call__(self, a):
    return (1/(np.sqrt(2*np.pi)*self.sig))*np.e**(-((a-self.u)**2)/(2*self.sig**2))

class SimpleNaiveBayes:
  class_probs = {}
  class_attr_gaussians = {}

  def __init__(self):
    pass

  def fit(self, X, y):
    self.class_probs = {
      lab:(np.count_nonzero(y == lab) / len(y)) 
      for lab in set(y)
    }
    self.class_attr_gaussians = {
      lab:[
        SimpleGaussian(
          np.mean(X[y == lab][:, i]),
          np.std(X[y == lab][:, i])
        )
        for i in range(X.shape[1])
      ] for lab in set(y)
    }
    return self

  def predict(self, X):
    return np.array([self.classify(x) for x in X])
  
  def classify(self, x):
    probs = []
    for k, p in self.class_probs.items():
      probs.append((
        p*(np.prod([self.class_attr_gaussians[k][i](a) for i, a in enumerate(x)])), k
      ))
    # TODO figure out why not reversing always gives the same results rather than inverting them
    return sorted(probs, reverse=True)[0][1]

if __name__ == "__main__":
  from sklearn.datasets import load_iris
  iris = load_iris()
  clf = SimpleNaiveBayes().fit(iris.data, iris.target)
  print(clf.predict(iris.data))