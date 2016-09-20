import numpy as np
from random import shuffle
import scipy

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[1]
  num_classes = W.shape[0]
  for i in xrange(num_train):
    f_scores = W.dot(X[:,i])
#    print f_scores.shape
#    print num_classes
#    assert f_scores.shape == (num_classes,)
    max_fscore = np.max(f_scores)
    f_scores -= max_fscore
    p_scores = np.exp(f_scores)
    pr_scores = p_scores / np.sum(p_scores)
    
    sum_fscores = np.sum(f_scores)
    sum_pscores = np.sum(p_scores)
#    loss -= f_scores[y[i]]
    loss -= np.log(p_scores[y[i]]/sum_pscores)
#    acc = 0.0
    for j in xrange(num_classes):
      if j == y[i]:
#        dW[j] -= X[:,i] * (1 - p_scores[j]/np.sum(p_scores))
#        dW[j,:] += X[:,i].T * ((p_scores[j]/sum_pscores) - 1.0)
        dW[j,:] += X[:,i].T * (1.0 - (p_scores[j]/sum_pscores))
        continue
#      acc += p_scores[j]
      # should it be minus?
#      dW[j] += X[:,i] * (p_scores[j]/np.sum(p_scores))
#      dW[j,:] += X[:,i].T * (p_scores[j]/sum_pscores)
      w_grad = X[:,i].T * (-p_scores[j]/sum_pscores)
#      print (p_scores[j]/sum_pscores)
#      print w_grad
#      print w_grad.shape
#      assert w_grad.shape == (1, X.shape[0])
      dW[j,:] += w_grad
#    loss += np.log(acc)
  
  dW *= -(1.0/num_train)
  loss /= num_train

  dW += reg * W
  loss += 0.5 * reg * np.sum(W * W)
    
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength

  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_classes = W.shape[0]
  num_features = W.shape[1]
  num_samples = X.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = np.dot(W, X)
  assert f.shape == (num_classes, num_samples)
  
  f_maxes = np.max(f, axis=0)
  shrunk_f = f - f_maxes
  e_shrunk_f = np.exp(shrunk_f)
  preds = e_shrunk_f / np.sum(e_shrunk_f, axis=0)

  assert f_maxes.shape == (num_samples,)
  assert shrunk_f.shape == (num_classes, num_samples)
  assert e_shrunk_f.shape == (num_classes, num_samples)
  assert preds.shape == (num_classes, num_samples)

  ij = np.array([[i, l] for i,l in enumerate(y)]).T
  ground_truth = scipy.sparse.coo_matrix((np.ones(num_samples), ij),
                                shape=(num_samples, num_classes)).todense().T
  assert ground_truth.shape == (num_classes, num_samples)

  loss = -1 * np.sum(np.multiply(ground_truth, np.log(preds))) / num_samples
  loss += (0.5 * reg) * np.sum(W*W)
  
  assert dW.shape == (num_classes, num_features)
  assert X.shape == (num_features, num_samples)
  assert ground_truth.shape == (num_classes, num_samples)
  dW = np.dot((ground_truth - preds), X.T) * (-1.0 / num_samples)
  dW += reg * W

  assert dW.shape == W.shape
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
