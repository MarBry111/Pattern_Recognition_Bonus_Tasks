#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def logistic_loss(X, y, w):
    """
    E = logistic_loss(X, y, w)

    Evaluates the logistic loss function.

    :param X:    d-dimensional observations, np.array (d, n)
    :param y:    labels of the observations, np.array (n, )
    :param w:    weights, np.array (d, )

    :return E:   calculated loss, python float
    """
    E = np.sum( np.log( 1 + np.exp( -w.dot(y*X) ) ) )/X.shape[1]
    return float(E)


def logistic_loss_gradient(X, y, w):
    """
    g = logistic_loss_gradient(X, y, w)

    Calculates gradient of the logistic loss function.

    :param X:   d-dimensional observations, np.array (d, n)
    :param y:   labels of the observations, np.array (n, )
    :param w:   weights, np.array (d, )
    :return g:  resulting gradient vector, np.array (d, )
    """
    g = np.sum( -y*X /(1 + np.exp(y*w.dot(X))), axis = 1 )/X.shape[1]
    return g


def logistic_loss_gradient_descent(X, y, w_init, epsilon):
    """
    w, wt, Et = logistic_loss_gradient_descent(X, y, w_init, epsilon)

    Performs gradient descent optimization of the logistic loss function.

    :param X:       d-dimensional observations, np.array (d, n)
    :param y:       labels of the observations, np.array (n, )
    :param w_init:  initial weights, np.array (d, )
    :param epsilon: parameter of termination condition: np.norm(w_new - w_prev) <= epsilon, python float
    :return w:      w - resulting weights, np.array (d, )
    :return wt:     wt - progress of weights, np.array (d, number_of_accepted_candidates)
    :return Et:     Et - progress of logistic loss, np.array (number_of_accepted_candidates, )
    """
    wt = w_init.reshape(w_init.shape[0],1)
    Et = np.array([logistic_loss(X, y, w_init)])
    
    step = 1.0
    w = wt[:,-1]
    g = logistic_loss_gradient(X, y, w)
    E = logistic_loss(X, y, w)

    while True:
        g_new = logistic_loss_gradient(X, y, w-step*g)
        E_new = logistic_loss(X, y, w-step*g)

        if np.sqrt(np.sum( np.abs(step*g)**2 )) < epsilon:
            w = w - step*g
            g = g_new
            E = E_new
            wt = np.append(wt, w.reshape(w.shape[0],1), axis = 1)
            Et = np.append(Et, E)
            return w, wt, Et
        elif E_new < E:
            w = w - step*g
            g = g_new
            E = E_new
            wt = np.append(wt, w.reshape(w.shape[0],1), axis = 1)
            Et = np.append(Et, E)
            step *= 2.0
        else:
            step /= 2.0

    return w, wt, Et


def classify_images(X, w):
    """
    y = classify_images(X, w)

    Classification by logistic regression.

    :param X:    d-dimensional observations, np.array (d, n)
    :param w:    weights, np.array (d, )
    :return y:   estimated labels of the observations, np.array (n, )
    """
    y = np.ones(X.shape[1])

    y[w.dot(X) < 0 ] = -1
    return y


def get_threshold(w):
    """
    thr = get_threshold(w)

    Returns the optimal decision threshold given the sigmoid parameters w

    :param w:    weights, np.array (d, )
    :return: calculated threshold (scalar)
    """
    thr = -w[0]/w[1]
    return thr

def logistic_regression_dim_lifting(X, y, w, fno = 7):
    x_min = np.min(X)
    x_max = np.max(X)

    #shifts
    alpha = np.linspace(x_min, x_max, fno).reshape(fno,1)+1 #(7,)
    aplha0 = 1.0

    tan_arr = []
    for i in range(fno):
        tan_arr.append( np.tanh(np.linspace(-5,5,100)*aplha0+ alpha[i]) )

    X = np.repeat(X[1,:].reshape(1, X.shape[-1]), 7, 0)
    
    X_t = np.append(np.ones((1, X.shape[-1])), np.tanh( X*aplha0 + alpha ), axis = 0)

    epsilon = 1e-2
    w, wt, Et = logistic_loss_gradient_descent(X_t, y, np.zeros(fno+1), epsilon)

    tan_arr = np.array(tan_arr)
    pk = w[1:].dot(tan_arr)
    return(pk, tan_arr, w, alpha, aplha0)