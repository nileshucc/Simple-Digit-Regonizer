# Simple-Digit-Regonizer
Implemented on Python using SVM


import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
digits = datasets.load_digits()
print(digits.data)
print(digits.target)
print(digits.images)
clf=svm.SVC(gamma=0.0000001,C=100) #increasing gamma value increases data processing but decreases accuracy.
X,y = digits.data[:-10], digits.target[:-10]
clf.fit(X,y)
print('Prediction=',clf.predict(digits.data[-5]))
plt.imshow(digits.images[-5], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
