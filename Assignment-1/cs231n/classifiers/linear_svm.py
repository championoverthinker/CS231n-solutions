import numpy as np
from random import shuffle
from past.builtins import xrange




def svm_loss_naive(W, X, y, reg):
 
  dW = np.zeros(W.shape) 
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    diff_count = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 
      if margin > 0:
        diff_count += 1
        dW[:, j] += X[i] 
        loss += margin
    
    dW[:, y[i]] += -diff_count * X[i]


  loss /= num_train
  dW /= num_train
  dW += reg*W 
  loss += 0.5 * reg * np.sum(W * W)
  return loss, dW




def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_score = scores[np.arange(num_train),y]
  margins = np.maximum(0,scores-correct_score[:,np.newaxis]+1.0)
  margins[np.arange(num_train),y] = 0
  loss = np.sum(margins)
  loss /= num_train
  loss += 0.5*reg * np.sum(W * W)
    
  pass
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

  mask = np.zeros(margins.shape)
  mask[margins>0] = 1
  incorrect_labels = np.sum(mask,axis=1)
  mask[np.arange(num_train),y] -= incorrect_labels
  dW = X.T.dot(mask)
  dW /= num_train
  dW += reg*W
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
