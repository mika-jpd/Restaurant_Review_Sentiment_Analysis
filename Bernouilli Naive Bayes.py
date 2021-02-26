import numpy
import pandas

class bernouilli_naive_bayes:

    def __init__(self, training, validation):
        self.attributes = {'Class prior':0, 'Theta_1':[], 'Theta_0':[], 'Results':[]}

        self.dataset_train = pandas.read_csv(training, header=None,delimiter="\t")
        self.dataset_validation = pandas.read_csv(validation, header=None, delimiter="\t")

        self.X = numpy.array(self.dataset_train.iloc[:, :-1])
        self.y = numpy.array(self.dataset_train.iloc[:, -1])

        self.xval = numpy.array(self.dataset_validation.iloc[:, :-1])
        self.yval = numpy.array(self.dataset_validation.iloc[:, -1])

    def fit(self):
        "Calculate class_prior"
        value = 0
        for i in self.y:
            if (i==1):
                value += 1
        value = value/len(self.y)
        self.attributes['Class prior'] = value

        "calculate positive and negative feature likelihoods"
        theta_j1 = [0]*len(self.X[0])
        theta_j0 = [0]*len(self.X[0])

        for i in range(0, len(self.X[0])):
            for j in range(0, len(self.X)):
                if(self.y[j] == 1):
                    theta_j1[i] += self.X[j][i]
                if(self.y[j] == 0):
                    theta_j0[i] += self.X[j][i]

        for i in range(0, len(theta_j1)):
            theta_j1[i] = theta_j1[i]/(value*len(self.y))
        for i in range(0, len(theta_j0)):
            theta_j0[i] = theta_j0[i]/(value*len(self.y))

        self.attributes['Theta_1'] = theta_j1
        self.attributes['Theta_0'] = theta_j0

    def save_parameters(self):
        cp = open("class_priors.tsv", "w+")
        cp.write(str(self.attributes['Class prior']) + "\n")
        cp.write(str(1-self.attributes['Class prior']) + "\n")

        nf = open("negative_feature_likelihoods.tsv", "w+")
        for i in self.attributes['Theta_0']:
            nf.write(str(i) + "\n")

        pf = open("positive_feature_likelihoods.tsv", "w+")
        for i in self.attributes['Theta_1']:
            pf.write(str(i) + "\n")

    def prediction(self, xtest, prior_prob_pos, prior_prob_neg, count_pos, count_neg):

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
    def accuracy(self, pred):
        return 100*numpy.mean(pred == self.yval)

x = bernouilli_naive_bayes("train_dataset.tsv", "validation_dataset.tsv")
x.fit()
x.save_parameters()

prior_data = numpy.array(pandas.read_csv("class_priors.tsv", header=None, delimiter="\t"))
prior_prob_pos = prior_data[0][0]
prior_prob_neg = prior_data[1][0]

count_pos = numpy.array(pandas.read_csv("positive_feature_likelihoods.tsv", header=None, delimiter="\t"))
count_neg = numpy.array(pandas.read_csv("negative_feature_likelihoods.tsv", header=None, delimiter="\t"))

ypred = x.prediction(x.xval, prior_prob_pos, prior_prob_neg, count_pos, count_neg)
print(x.accuracy(ypred))
