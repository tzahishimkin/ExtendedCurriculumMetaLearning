import numpy
from cvxopt import matrix, solvers
import math
import scipy.special

from main_seq_learning import fixed_order, learned_order, learned_order_extended, predict, test_error

def getdata(tasknames):
   
    data = []; labels = []; test_data = []; test_labels = []
    
    for i,name in enumerate(tasknames):

        X=numpy.loadtxt("data/%s_train.txt"%name)
        X = numpy.c_[X,[1]*len(X)] 

        testX = numpy.loadtxt("data/%s_test.txt"%name)    
        testX = numpy.c_[testX,[1]*len(testX)]  

        Y = numpy.r_[[+1]*50,[-1]*50]
        testY = numpy.r_[[+1]*50,[-1]*50]

        data.append(X)
        labels.append(Y)
        test_data.append(testX)
        test_labels.append(testY)

    return data,labels,test_data,test_labels

            
def main():

    tasknames = numpy.asarray(['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'], dtype='str')
    [data, labels, test_data, test_labels]=getdata(tasknames)
    
    numpy.random.seed(1)

    print('***Learning the tasks in a data-dependent order')
    print('***SeqMT algorithm')
    [weights, C, order] = learned_order(data, labels)
    predictions = predict(test_data, weights)
    error_learned = test_error(test_labels, predictions)
    print "Avg. error rate across tasks: %.2f"%(numpy.mean(error_learned)*100.)
    print "Learned order of tasks:", tasknames[order]

    print('***MultiSeqMT algorithm')
    [weights, C, tree] = learned_order_extended(data, labels)
    predictions = predict(test_data, weights)
    error_ext = test_error(test_labels, predictions)
    #print "Learned order of tasks:", tree
    print "Avg. error rate across tasks: %.2f"%(numpy.mean(error_ext)*100.)

    print('***As a reference') 
    print('***Learning the tasks in a random order') 
    rand_order = numpy.random.permutation(10)
    [weights, C] = fixed_order(data, labels, rand_order)
    predictions = predict(test_data, weights)
    error_random = test_error(test_labels, predictions)
    print "Avg. error rate across tasks: %.2f"%(numpy.mean(error_random)*100.)
    print "Random order of tasks:", tasknames[rand_order]
    
    return 1

if __name__ == '__main__':

    main()
