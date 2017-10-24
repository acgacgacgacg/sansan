import numpy as np
import cv2
import csv
import warnings
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.neural_network import BernoulliRBM

# HOG as feature extraction, can be combined into data pipeline
def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    h, v, c = np.shape(bins)
    bin_cells = [bins[j-v/2:j, h-h/8:h, :] for i in xrange(h/4, h, h/4) for j in xrange(v/2, v, v/2)]
    mag_cells = [mag[j-v/2:j, h-h/8:h, :] for i in xrange(h/4, h, h/4) for j in xrange(v/2, v, v/2)]

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist

# Try PCA as dimension reduction, SVC with Gaussian Kernel as classifier
def ParaTun(X_dev, Y_dev):
	steps = [('scaler', StandardScaler()),
	         ('decomposer', PCA(random_state=0)),
	         ('classifier', OneVsRestClassifier(SVC(kernel='rbf', gamma=0.0005, probability=True)))]
	pipeline = Pipeline(steps)
	params = {'classifier__estimator__C': [10],
	         'decomposer__n_components': [100],
	         'classifier__estimator__gamma': [0.0005]}
	#scorer = make_scorer(roc_auc_score, average='macro', needs_proba=True)
	print '1'
	predictor = GridSearchCV(pipeline, params, n_jobs=1)
	#predictor = GridSearchCV(pipeline, params, n_jobs=1)

	print '2'
	result = predictor.fit(X_dev, Y_dev)
	print result.best_score_
	#print result.cv_results_
	print result.best_params_
	return predictor

# Try RBM as feature extraction and linearSVC as classifier
def ParaTun2(X_dev, Y_dev):
	rbm = BernoulliRBM(random_state=0, verbose=True)
	steps = [('rbm', rbm),
	         ('classifier', OneVsRestClassifier(LinearSVC()))]
	rbm.learning_rate = 0.005
	rbm.n_iter = 200
	rbm.n_components = 100
	#rbm.batch_size = 10
	pipeline = Pipeline(steps)
	params = {'classifier__estimator__C': [10]}
	#scorer = make_scorer(roc_auc_score, average='macro', needs_proba=True)
	predictor = GridSearchCV(pipeline, params, cv=2,  n_jobs=1)
	#predictor = GridSearchCV(pipeline, params, n_jobs=1)

	print '2'
	result = predictor.fit(X_dev, Y_dev)
	print result.best_score_
	#print result.cv_results_
	print result.best_params_
	return predictor

# Transfer image to binary data, prepare for RBM
def bitWise(arrImg):
	img = arrImg.reshape((299, 50))
	#img = cv2.GaussianBlur(img,(3,3),0);
	img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,7)
	img = cv2.bitwise_not(img, img)
	#img = cv2.GaussianBlur(img, (3,3), 0)
	return img.flatten()/255


path = '~/'
X = np.load(path+'trainDataH.npy')
Y_dev = np.load(path+'labelsH.npy')
X_dev = cv2.cvtColor(X, cv2.COLOR_BGR2GRAY)
X_dev = np.array([bitWise(img) for img in X_dev])
clfH = ParaTun(X_dev, Y_dev)
joblib.dump(clfH, path+'SVMH.pkl') 






