#!/usr/bin/python
from sklearn import tree

# this is used to classified fruits between orange (1) and apple (0)
# first we need to give training dataset (features: smooth is 1 while bumpy is 0) and provide the labels
featuresT = [[150, 0], [170, 0], [130,1], [140, 1]]
labelT = [1, 1, 0, 0]

clf = tree.DecisionTreeClassifier()
clf.fit(featuresT, labelT)

print(clf.predict([[150, 0]]))

# the next example is a bit complicated! we use sklean dataset iris
# the package will provide meta-data to tell you the feature and labels (here refer to as target)
from sklearn.datasets import load_iris

iris = load_iris()
print (iris.feature_names)
print (iris.target_names)
print (iris.data[0])        # first sample
print (iris.target[0])      # first label

# to print the entire dataset, you could use the following
#for i in range(len(iris.data)):
#    print ("Example %d: label %s, and features %s" % (i, iris.target[i], iris.data[i]))

# here we need to split the data into training and testing subsets    
import numpy as np

test_idx =[0, 50, 100]          # the first item for each types of flower

# training set
training_data = np.delete(iris.data, test_idx, axis=0)
training_target = np.delete(iris.target, test_idx)

# testing set
test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

# using decision tree classifier to do the training
clf.fit(training_data, training_target)

print(test_target)
print(clf.predict(test_data))

# now you could try to output the tree visually
from sklearn.externals.six import StringIO
import pydotplus, os

os.sys.path.append(r'/usr/include/X11/bitmaps/dot')

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=iris.feature_names, class_names=iris.target_names,
                        filled=True, rounded=True, special_characters=True, impurity=False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

#graph.write_pdf(r'iris.pdf')
#Image(graph.create_png())


X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.5)

# use the previous tree imported for the classification
#clf2 = tree.DecisionTreeClassifier()

# or you could use a different algorithm for classification
from sklearn.neighbors import KNeighborsClassifier
clf2 = KNeighborsClassifier()

clf2.fit(X_train, y_train)
predictions = clf2.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))


