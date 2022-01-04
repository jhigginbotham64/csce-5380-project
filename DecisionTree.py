from scipy.stats import mode
import numpy as np

class SimpleCondition:
  func = None
  outcomes = None

  def __init__(self, func, outcomes):
    self.func = func
    self.outcomes = outcomes
  
  def __call__(self, x):
    return self.func(x)
  
  def __lt__(self, other):
    # needs to be implemented so that i can sort
    # arrays of tuples of gini indices and their
    # corresponding conditions.
    return hash(self) < hash(other)

class SimpleDecisionTreeNode:
  label = None
  condition = None
  children = {}
  
  def __init__(self, condition = None, label = None):
    self.label = label
    self.condition = condition

  def isleaf(self):
    return self.label is not None
  
  def add_child(self, outcome, N):
    self.children[outcome] = N
  
  def outcomes(self):
    return self.children.keys()
  
  def traverse(self, x):
    if self.isleaf():
      return self.label
    return self.children[self.condition(x)].traverse(x)

class SimpleDecisionTree:
  root = None
  
  def __init__(self):
    pass

  def fit(self, X, y):
    self.root = SimpleDecisionTree.gentree(X, y)
    return self

  @staticmethod
  def gini(X, y):
    return 1 - sum([(
        np.count_nonzero(y == c) / len(y)) ** 2 
        for c in set(y)])

  @staticmethod
  def gini_ind_cont(m, X, y):
    le_inds = X <= m
    g_inds = X > m
    g1 = np.count_nonzero(le_inds) * SimpleDecisionTree.gini(X[le_inds], y[le_inds])
    g2 = np.count_nonzero(g_inds) * SimpleDecisionTree.gini(X[g_inds], y[g_inds])
    return (g1 + g2) / len(X)


  @staticmethod
  def attribute_selection(X, y):
    conds = []
    for col in range(X.shape[-1]):
      # https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
      inds = X[:, col].argsort()
      sortedX = X[inds][:, col]
      sortedy = y[inds]
      mids = []
      for i in range(len(sortedX) - 1):
        mids.append((sortedX[i] + sortedX[i + 1]) / 2)
      gini_inds = []
      for m in mids:
        gini_inds.append(SimpleDecisionTree.gini_ind_cont(m, sortedX, sortedy))
      g, m = sorted(list(zip(gini_inds, mids)))[0]
      conds.append((g, SimpleCondition(func=lambda x: x[col] <= m, outcomes=[True, False])))
    return sorted(conds)[0][1]
      
  @staticmethod
  def gentree(X, y):
    n = SimpleDecisionTreeNode()
    majority = mode(y).mode[0]
    if len(set(y)) == 1:
      n.label = majority
      return n
    # NOT dealing with discrete-valued attributes here
    cond = SimpleDecisionTree.attribute_selection(X, y)
    n.condition = cond
    for j in cond.outcomes:
      inds = [cond(x) == j for x in X]
      Xj = X[inds]
      if len(Xj) == 0:
        n.add_child(j, SimpleDecisionTreeNode(label=majority))
      else:
        n2 = SimpleDecisionTree.gentree(Xj, y[inds])
        n.add_child(j, n2)
    return n

  def traverse(self, x):
    return self.root.traverse(x)

  def predict(self, X):
    return np.array([self.traverse(x) for x in X])

if __name__ == "__main__":
  from sklearn.datasets import load_iris
  iris = load_iris()
  clf = SimpleDecisionTree().fit(iris.data, iris.target)
  print(clf.predict(iris.data))