import numpy


class bernouilli_naive_bayes:
    attributes = {'Class prior':0, 'Theta_1':[], 'Theta_0':[]}

    dataset_train = numpy.loadtxt(
        "C:/Users/mikad/Desktop/Winter 2021/Comp 451/Assignements/Assignement 2/practical_assn_1/practical_assn_1/train_dataset.tsv",
        delimiter="\t")

    dataset_test = numpy.loadtxt(
        "C:/Users/mikad/Desktop/Winter 2021/Comp 451/Assignements/Assignement 2/practical_assn_1/practical_assn_1/validation_dataset.tsv",
        delimiter="\t")

    X = dataset_train[:, :-1]
    y = dataset_train[:, -1]

    def fit(features, labels):

        value = 0
        for i in y:
            if (i==1):
                value += 1
        value = value/len(y)
        attributes['Class prior'] = value

        print(attributes['Class prior'])

        theta_j1 = [0]*len(X[0])
        theta_j0 = [0]*len(X[0])

        for i in range(0, len(X)):
            for x in range(0, len(X[i])):
                if((X[i][x] == 1) & (y[i] == 1)):
                    theta_j1[x] += 1
                if ((X[i][x] == 1) & (y[i] == 0)):
                    theta_j0[x] += 1

        for x in range(0, len(theta_j1)):
            theta_j1[x] = theta_j1[x]/len(theta_j1)
            theta_j0[x] = theta_j0[x]/len(theta_j0)
        attributes['Theta_1'] = theta_j1
        attributes['Theta_0'] = theta_j0

    def predict(dataset_test, attributes):
        X = dataset_test[:, :-1]
        y = dataset_test[:, -1]

        c = attributes['Class prior']
        theta_1 = attributes['Theta_1']
        theta_0 = attributes['Theta_0']

        result = [0]*len(dataset_test)
        value_1 = 1
        value_2 = 1

        for i in range(0, len(X)):
            for j in range(0, len(X[i])):
                value_1 = value_1*c*(theta_1[j]**(X[i][j]))*(1-theta_1[j])**(1-X[i][j])
                value_2 = value_2*(1-c)*(theta_0[j]**(X[i][j]))*(1-theta_0[j])**(1-X[i][j])
            print(value_1, value_2)
            result[i] = value_1/value_2

