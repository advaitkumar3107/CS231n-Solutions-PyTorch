from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_training = X.shape[0]
    num_classes = W.shape[1]
    score = np.matmul(X, W)
    
    for i in range(num_training):
        stable = score[i] - np.max(score[i])
        loss_function = np.exp(stable)/np.sum(np.exp(stable))
        loss = loss - np.log(loss_function[y[i]])
        
        for j in range(num_classes):
            dW[:,j] = dW[:,j] + X[i]*loss_function[j]
        dW[:,y[i]] = dW[:,y[i]] - X[i]
    
    loss = loss/num_training
    dW = dW/num_training
    dW = dW + 2 * reg * W
    loss = loss + reg * np.sum(W**2)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_training = X.shape[0]
    score = np.matmul(X, W)
    score = score - np.max(score, axis = 1, keepdims = True)
    denominator = np.sum(np.exp(score), axis = 1, keepdims = True)
    numerator = np.exp(score[range(num_training), y].reshape(num_training, 1))
    loss_term = np.divide(numerator, denominator)
    
    loss = loss - np.sum(np.log(loss_term))
    loss = loss/num_training
    loss = loss + reg * np.sum(W**2)
    
    denominator = np.sum(np.exp(score), axis = 1, keepdims = True)
    term = np.exp(score)/denominator
    term[range(num_training), y] -= 1
    dW = np.matmul(np.transpose(X) , term)
    dW = dW/num_training
    dW = dW + 2*reg*W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
