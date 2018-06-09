import numpy
import math
import scipy.special
from cvxopt import matrix, solvers
import pdb
solvers.options['show_progress'] = False

# List of all functions:
# ASVM - Adaptive SVM with fixed parameters
# ASVM_CV - Adaptive SVM with 5x5 cross validation to tune C
# fixed_order - sequence of Adaptive SVMs solved in a pre-defined fixed order (for example, in a random order)
# learned_order - SeqMT algorithm that learns a data-dependent order of tasks based on a greedy minimisation of the bound
# learned_order_extended - MultiSeqMT algorithm that learns a data-dependent order of tasks with multiple subsequences
# test_error - returns the error rate for all tasks
# predict - returns a dictionary of predicted scores for all tasks

# List of all parameters:
# data - list of data matrices for n tasks
# labels - list of label vectors for n tasks
# X - data for one task, which is a matrix of size nxd with n data points of dimensionality d
# Y - labels for one task, which is a vector of size n with binary labels
# w_source - source weight vector of dimensionality d
# C - regularization parameter in Adaptive SVM (in front of loss, divided by the number of points)
# method - ASVM method can be solved using QP in 'primal' or 'dual' variables; default is 'dual'
# order - order of tasks, which is a vector of tasks indices (from 0 to n-1)
# tree - multiple subsequences of tasks are represented as a tree structure. 
#        It is an nx2 matrix, where (i,j) indicates that j-th task used the outcome of the i-th task as a source. 
#        If i=-1, it means that j-th task is a beginning of a new subsequence, i.e. it was solved using standard SVM without adaptation.
#        j-th task means task with j-th index, and tasks indices range from 0 to n-1. 

#
########Adaptive SVM
#
def ASVM(X, Y, w_source, C, method='dual'):

    m = numpy.size(X,0)
    deltas = 1. - Y*numpy.dot(X, w_source)

    if (method == 'primal'):
        w_hat = ASVM_primal(X, Y, deltas, C)

    if (method == 'dual'):
        w_hat = ASVM_dual(X, Y, deltas, C)
    
    w = w_hat + w_source

    return w

def ASVM_CV(X, Y, w_source, method='dual'):

    #model selection over a range of values 
    pos_C = [0.01, 0.1, 1., 10., 100., 1000., 10000., 100000.]	
    l = len(pos_C)
    cv_acc = numpy.zeros(l)
    n = numpy.size(X, 0)
    
    numpy.random.seed(1)#seed

    for cv_it in range(5):
        allind = numpy.arange(n)
        mask = Y == 1
        ind_pos = numpy.random.permutation(allind[mask[:]])
        ind_neg = numpy.random.permutation(allind[~mask[:]])
        ind = numpy.hstack((ind_pos, ind_neg))
        for split in range(5):
            test_ind = ind[split::5]
            mask = numpy.ones(ind.shape, dtype = bool)
            mask[test_ind[:]] = 0
            train_ind = ind[mask]
            test_data = X[test_ind[:],:]
            test_labels = Y[test_ind[:]]
            train_data = X[train_ind[:],:]
            train_labels = Y[train_ind[:]]
            for it in range(l):
                C = pos_C[it]
                w = ASVM(train_data, train_labels, w_source, C, method)
                predictions = numpy.dot(test_data, w)
                cv_acc[it] = cv_acc[it] + numpy.sum(predictions*test_labels > 0)*1./numpy.size(test_data, 0)
    cv_acc = cv_acc/25
    opt = numpy.where(cv_acc == numpy.max(cv_acc))
    opt_C = pos_C[opt[0][0]]
    w = ASVM(X, Y, w_source, opt_C, method)
        
    return  [w, opt_C]


def ASVM_primal(X, Y, deltas, C1):

    d = numpy.size(X,1)
    m = numpy.size(X,0)

    P = numpy.zeros((d+m, d+m))
    q = numpy.zeros(d+m)
    G = numpy.zeros((2*m, d+m))
    h = numpy.zeros(2*m)
    
    P[0:d,0:d] = 2*numpy.eye(d)
    q[d:d+m] = (C1/m)*numpy.ones(m)
    G[0:m,0:d] = -numpy.dot(numpy.diag(Y), X)
    G[0:m,d:d+m] = -numpy.eye(m)
    G[m:2*m, d:d+m] = -numpy.eye(m)
    h[0:m] = -deltas

    solution = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), None, None)['x']
    w = numpy.squeeze(numpy.array(solution[0:d]))

    return w

def ASVM_dual(X, Y, deltas, C1):

    m = numpy.size(X,0)

    P = numpy.dot(numpy.diag(Y), numpy.dot(numpy.dot(X, numpy.transpose(X)), numpy.diag(Y)))/2
    q = -deltas
    G = numpy.zeros((2*m,m))
    G[0:m,:] = - numpy.eye(m)
    G[m:2*m,:] = numpy.eye(m)
    h = numpy.zeros(2*m)
    h[m:2*m] = (C1/m)*numpy.ones(m)

    alphas = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), None, None)['x']
    alphas = numpy.array(alphas[0:m])

    w = numpy.dot(numpy.squeeze(alphas), numpy.dot(numpy.diag(Y), X))/2

    return w


#
########Learning the tasks in a fixed order
#
def fixed_order(data, labels, order, method='dual'):
    
    n = len(data)
    d = numpy.size(data[0], 1)
    weights = {}
    Cs = numpy.zeros(n)
    w_source = numpy.zeros(d)
    
    for i in order:

        [w, Cs[i]] = ASVM_CV(data[i], labels[i], w_source, method)
        w_source = w
        weights[i] = w
    
    return [weights, Cs]

#
########Learning the tasks in a data-dependent order (SeqMT algorithm)
#
def learned_order(data, labels, method='dual'):
    
    n = len(data)
    d = numpy.size(data[0],1)
    order = numpy.array(range(n))
    weights = {}
    Cs = numpy.zeros(n)
    
    m_bar = 0
    for i in range(n):
        m = numpy.size(data[i], 0)
        m_bar = m_bar + 1./m
    m_bar = m_bar/n
    m_bar = 1./m_bar

    w_source = numpy.zeros(d)
    for i in range(n):

        opt_task_ind = -1
        opt_objective = 0
        
        for j in range(i,n):
        
            X = data[order[j]]
            Y = labels[order[j]]
            [w, qwe] = ASVM_CV(X, Y, w_source, method)

            # compute error part of the bound
            emp_loss = 0
            for k in range(numpy.size(X,0)):
                z = Y[k]*numpy.dot(X[k,:], w)/numpy.linalg.norm(X[k,:])
                Phi = 0.5*(1 - scipy.special.erf(z/math.sqrt(2)))
                emp_loss = emp_loss + Phi
            emp_loss = emp_loss/numpy.size(X,0)

            # compute regularization part of the bound
            reg = (numpy.linalg.norm(w-w_source)**2)/(2*math.sqrt(m_bar))
            objective = emp_loss + reg
            if (opt_task_ind == -1) or (objective < opt_objective): #If first interation or the objective is minimal
                opt_task_ind = j
                opt_objective = objective
        
        X = data[order[opt_task_ind]]
        Y = labels[order[opt_task_ind]]
        [w, Cs[order[opt_task_ind]]] = ASVM_CV(X, Y, w_source, method)
        weights[order[opt_task_ind]] = w
        w_source = w
        q = order[i]
        order[i] = order[opt_task_ind]
        order[opt_task_ind] = q
    
    return [weights, Cs, order]

#
########Learning multiple subsequences of data-dependent order of tasks (MultiSeqMT algorithm)
#
def learned_order_extended(data, labels, method='dual'):

    n = len(data)
    d = numpy.size(data[0],1)
    order = numpy.array(range(n))
    tree = numpy.zeros((n, 2))
    weights = {}
    Cs = numpy.zeros(n)

    m_bar = 0
    for i in range(n):
        m = numpy.size(data[i], 0)
        m_bar = m_bar + 1./m
    m_bar = m_bar/n
    m_bar = 1./m_bar

    w_source = numpy.zeros(d)
    sources = {-1: w_source}

    for i in range(n):

        opt_task_ind = -1
        opt_source_ind = -1
        opt_objective = 0

        for j in range(i,n):

            X = data[order[j]]
            Y = labels[order[j]]
            for source_ind, w_source in sources.iteritems():
                [w, qwe] = ASVM_CV(X, Y, w_source, method)

                # compute error part of the bound
                emp_loss = 0
                for k in range(numpy.size(X,0)):
                    z = Y[k]*numpy.dot(X[k,:], w)/numpy.linalg.norm(X[k,:])
                    Phi = 0.5*(1 - scipy.special.erf(z/math.sqrt(2)))
                    emp_loss = emp_loss + Phi
                emp_loss = emp_loss/numpy.size(X,0)

                # compute regularization part of the bound
                reg = (numpy.linalg.norm(w-w_source)**2)/(2*math.sqrt(m_bar))
                objective = emp_loss + reg
                #pdb.set_trace()
                if (opt_task_ind == -1) or (objective < opt_objective):
                    opt_task_ind = j
                    opt_source_ind = source_ind
                    opt_objective = objective

        X = data[order[opt_task_ind]]
        Y = labels[order[opt_task_ind]]
        w_source = sources[opt_source_ind]
        [w, Cs[order[opt_task_ind]]] = ASVM_CV(X, Y, w_source, method)
        weights[order[opt_task_ind]] = w
        tree[i,0] = opt_source_ind
        tree[i,1] = order[opt_task_ind]
        if (opt_source_ind != -1):
            del sources[opt_source_ind]
        sources[order[opt_task_ind]] = w
        q = order[i]
        order[i] = order[opt_task_ind]
        order[opt_task_ind] = q

    return [weights, Cs, tree]

#
########Performance
#
def test_error(labels, predictions):

    n = len(labels)
    error = numpy.zeros(n)
    for ind, pred_scores in predictions.iteritems():
        error[ind] = numpy.sum(labels[ind]*pred_scores <= 0)*1./numpy.size(labels[ind],0)

    return error

def predict(data, weights):

    predictions = {}
    for ind, w in weights.iteritems():
        predictions[ind] = numpy.dot(data[ind], w)

    return predictions

