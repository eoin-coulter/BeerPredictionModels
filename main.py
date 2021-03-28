# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


#load data to csv
#add columns
#


















import graphviz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC




featuredCols = ['calorific_value', 'nitrogen', 'turbidity', 'alcohol', 'sugars', 'bitterness', 'beer_id', 'colour',
                'degree_of_fermentation']

colNames = ['number', 'alcohol_by_weight', 'rating', 'bitterness', 'nitrogen', 'turbidity', 'sugars', 'degree_of_fermentation', 'calorific_value',
            'density', 'pH','colour','sulphites']

trainingData = pd.read_csv('beer_training.txt', sep='\t', names=colNames)
testData = pd.read_csv('beer_test.txt', sep='\t', names=colNames)
def id3method():
    print('hi')
    print(trainingData)





    X_train = trainingData[featuredCols]
    Y_train = trainingData.BEERStyle
    X_test = testData[featuredCols]
    Y_test = testData.BEERStyle




    clf = DecisionTreeClassifier(criterion = "entropy")
    clf = clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))


    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=X_test.columns,class_names=clf.classes_)
    graph = graphviz.Source(dot_data)
    graph.render("beer")

    gb = GaussianNB()
    gb = gb.fit(X_train, Y_train)
    Y_pred = gb.predict(X_test)




    print("Accuracy:", metrics.accuracy_score(Y_test, Y_pred))
    print(metrics.classification_report(Y_test, Y_pred))
    print(metrics.confusion_matrix(Y_test, Y_pred))

    plt.show()


    




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    id3method()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
