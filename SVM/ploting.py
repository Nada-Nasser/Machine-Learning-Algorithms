from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from mpl_toolkits.mplot3d import Axes3D
iris = datasets.load_iris()
X = iris.data[:, :3] # we only take the first three features.
Y = iris.target
#make it binary classification problem
X = X[np.logical_or(Y==0,Y==1)]
Y = Y[np.logical_or(Y==0,Y==1)]
model = svm.SVC(kernel='linear')
clf = model.fit(X, Y)
# The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
# Solve for w3 (z)
z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]
tmp = np.linspace(-100,100,30)
x,y = np.meshgrid(tmp,tmp)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'ob')
ax.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'sr')
ax.plot_surface(x, y, z(x,y))
ax.view_init(30, 60)
plt.show()

# w = []
# f = lambda x,y: (-w[0] - w[1]*x - w[2] *y) / w[3]
#
# tmp = np.linspace(-100,100,30)
# xxx,yyy = np.meshgrid(tmp,tmp)
# fig3 = plt.figure()
# axx = fig3.add_subplot(111, projection='3d')
#
# axx.plot3D(X[Y==0,0], X[Y==0,1], X[Y==0,2],'ob')
# axx.plot3D(X[Y==1,0], X[Y==1,1], X[Y==1,2],'sr')
#
# axx.plot_surface(xxx, yyy, f(xxx,yyy))
# axx.view_init(30, 60)
# plt.show()