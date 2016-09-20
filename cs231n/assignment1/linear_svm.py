import numpy as np
from random import shuffle
import math

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[0]
  num_train = X.shape[1]
  loss = 0.0

  for i in xrange(num_train):
    scores = W.dot(X[:, i])
    correct_class_score = scores[y[i]]
    dW_yi = 0.0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[j,:] += X[:,i]
        dW_yi += X[:,i]
    dW[y[i],:] -= dW_yi
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
#  loss += 0.5 * reg * np.sum(W * W)
  loss += 0.5 * reg * math.pow(np.linalg.norm(W, 2), 2)
  dW += (reg * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_classes = W.shape[0]
  num_train = X.shape[1]
  num_features = W.shape[1]

  assert dW.shape == (num_classes, num_features)
  assert X.shape == (num_features, num_train)

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = W.dot(X)
  assert scores.shape == (num_classes, num_train)
  
  # broadcast correct score
  # 1 x num_train
  y_scores = np.array([scores[correct_class, train_num] for train_num, correct_class in enumerate(y)])
  margin = scores - y_scores + 1

  loss = np.sum(np.maximum(np.zeros(margin.shape), margin)) - num_train

  assert margin.shape == (num_classes, num_train)

  # this is just for incorrect classes
  dW = np.dot(((margin -1 ) > -1), X.T)

  # to compute adjustment for correct class:
  #   the adjustment for each train sample's feature is -(num_violated_margin * feature_val)
  #   mask out the correct class's X values and the ones that don't cross the margin
  #   sum on each
  num_violated = np.sum(((margin-1) > -1), axis=0)
  assert num_violated.shape == (num_train, )
  correct_class_adj = X * num_violated
  
  assert correct_class_adj.shape == X.shape

  dW -= np.dot(((margin -1) == 0), correct_class_adj.T)
  
  
#  correct_cls_oj = np.array([scores[correct_class, train_num] for train_num, correct_class in enumerate(y)])
  assert dW.shape == (num_classes, num_features)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * math.pow(np.linalg.norm(W, 2), 2)
#  loss += 0.5 * reg * np.sum(W * W)
  dW += (reg * W)
  
  return loss, dW
