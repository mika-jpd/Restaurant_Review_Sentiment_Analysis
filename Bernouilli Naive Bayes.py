import numpy
import pandas

class bernouilli_naive_bayes:
    attributes = {'Class prior':0, 'Theta_1':[], 'Theta_0':[], 'Results':[]}

    dataset_train = numpy.loadtxt(
        "train_dataset.tsv",
        delimiter="\t")

    dataset_test = numpy.loadtxt(
        "validation_dataset.tsv",
        delimiter="\t")

    X = dataset_train[:, :-1]
    y = dataset_train[:, -1]

    val = pandas.read_csv("validation_dataset.tsv", header=None, delimiter="\t")

    xval = numpy.array(val.iloc[:, :-1])
    yval = numpy.array(val.iloc[:, -1])

    def fit(X, y, attributes):

        value = 0
        for i in y:
            if (i==1):
                value += 1
        value = value/len(y)
        attributes['Class prior'] = value

        print(attributes['Class prior'])

        theta_j1 = [0]*len(X[0])
        theta_j0 = [0]*len(X[0])

        for i in range(0, len(X[0])):
            for j in range(0, len(X)):
                if(y[j] == 1):
                    theta_j1[i] += X[j][i]
                if(y[j] == 0):
                    theta_j0[i] += X[j][i]

        for i in range(0, len(theta_j1)):
            theta_j1[i] = theta_j1[i]/(value*len(y))
        for i in range(0, len(theta_j0)):
            theta_j0[i] = theta_j0[i]/(value*len(y))

        attributes['Theta_1'] = theta_j1
        attributes['Theta_0'] = theta_j0




    def save_parameters(attributes):
        cp = open("class_priors.tsv", "w+")
        cp.write(str(attributes['Class prior']) + "\n")
        cp.write(str(1-attributes['Class prior']) + "\n")

        nf = open("negative_feature_likelihoods.tsv", "w+")
        for i in attributes['Theta_0']:
            nf.write(str(i) + "\n")

        pf = open("positive_feature_likelihoods.tsv", "w+")
        for i in attributes['Theta_1']:
            pf.write(str(i) + "\n")

    def prediction(xtest, prior_prob_pos, prior_prob_neg, count_pos, count_neg):

        ypred = numpy.zeros(xtest.shape[0])
        log_likelihood_pos = 0
        log_likelihood_neg = 0
        total_pos = 0
        total_neg = 0

        log_likelihood_pos = numpy.matmul(xtest, numpy.log(count_pos)) + numpy.matmul((numpy.ones(xtest.shape) - xtest), numpy.log(
            numpy.ones(count_pos.shape) - count_pos))

        log_likelihood_neg = numpy.matmul(xtest, numpy.log(count_neg)) + numpy.matmul((numpy.ones(xtest.shape) - xtest), numpy.log(
            numpy.ones(count_neg.shape) - count_neg))

        total_pos = numpy.log(prior_prob_pos) * numpy.ones(log_likelihood_pos.shape) + log_likelihood_pos
        total_neg = numpy.log(prior_prob_neg) * numpy.ones(log_likelihood_neg.shape) + log_likelihood_neg

        for i in range(total_pos.shape[0]):

            if total_pos[i] >= total_neg[i]:

                ypred[i] = 1

            elif total_pos[i] < total_neg[i]:

                ypred[i] = 0

        return ypred

    if __name__ == '__main__':
        fit(X, y, attributes)
        save_parameters(attributes)

        count_pos = numpy.array(pandas.read_csv("positive_feature_likelihoods.tsv", header=None, delimiter="\t"))
        count_neg = numpy.array(pandas.read_csv("negative_feature_likelihoods.tsv", header=None, delimiter="\t"))


        label_pred = prediction(xval, attributes['Class prior'], 1 - attributes['Class prior'], count_pos, count_neg)
        print(label_pred)

