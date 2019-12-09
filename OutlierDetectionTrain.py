###
### Original source: https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/
###
### Modified and redisgned to fit an anomaly detection ROS topic for a
### fault tolerant robot for CPE 446 at Cal Poly SLO.
###
### Authors: Lakshay Arora, Kevin Dixson, Sukhman Marok
### 

import time

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
import pandas as pd


# Import models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF

from pyod.utils.data import generate_data, get_outliers_inliers

# generate random data with two features
'''
n_train : int, (default=1000)
        The number of training points to generate.

train_only : bool, optional (default=False)
        If true, generate train data only.

n_features : int, optional (default=2)
        The number of features (dimensions).
'''


df = pd.read_csv("train_bad.csv")

df.plot.scatter('Item_MRP', 'Item_Outlet_Sales')

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
df[['Item_MRP','Item_Outlet_Sales']] = scaler.fit_transform(df[['Item_MRP','Item_Outlet_Sales']])
df[['Item_MRP','Item_Outlet_Sales']].head()
X1 = df['Item_MRP'].values.reshape(-1, 1)
X2 = df['Item_Outlet_Sales'].values.reshape(-1, 1)

X = np.concatenate((X1, X2), axis=1)

random_state = np.random.RandomState(42)
outliers_fraction = 0.01
# Define seven outlier detection tools to be compared
classifiers = {
        'Angle-based Outlier Detector (ABOD-5)': ABOD(contamination=outliers_fraction, n_neighbors=5),
        #'Angle-based Outlier Detector (ABOD-25)': ABOD(contamination=outliers_fraction, n_neighbors=25),
        #'Angle-based Outlier Detector (ABOD-50)': ABOD(contamination=outliers_fraction, n_neighbors=50),
        #'Angle-based Outlier Detector (ABOD-100)': ABOD(contamination=outliers_fraction, n_neighbors=100),
        #'Angle-based Outlier Detector (ABOD-500)': ABOD(contamination=outliers_fraction, n_neighbors=500),
        #'Angle-based Outlier Detector (ABOD-8000)': ABOD(contamination=outliers_fraction, n_neighbors=8000),
        #'Cluster-based Local Outlier Factor (CBLOF)': CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state),
        #'Feature Bagging': FeatureBagging(LOF(n_neighbors=35), contamination=outliers_fraction,check_estimator=False, random_state=random_state),
        #'Histogram-base Outlier Detection (HBOS)': HBOS(contamination=outliers_fraction),
        'Isolation Forest': IForest(contamination=outliers_fraction, random_state=random_state, behaviour="new"),
        'K Nearest Neighbors (KNN-5)': KNN(contamination=outliers_fraction, n_neighbors=5),
        'K Nearest Neighbors (KNN-25)': KNN(contamination=outliers_fraction, n_neighbors=25),
        'K Nearest Neighbors (KNN-50)': KNN(contamination=outliers_fraction, n_neighbors=50),
        #'K Nearest Neighbors (KNN-100)': KNN(contamination=outliers_fraction, n_neighbors=100),
        #'K Nearest Neighbors (KNN-500)': KNN(contamination=outliers_fraction, n_neighbors=500),
        #'K Nearest Neighbors (KNN-8000)': KNN(contamination=outliers_fraction, n_neighbors=8000),
        #'Average KNN': KNN(method='mean',contamination=outliers_fraction)
}

xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))

for i, (clf_name, clf) in enumerate(classifiers.items()):
    start = time.time()
    clf.fit(X)
    # predict raw anomaly score
    scores_pred = clf.decision_function(X) * -1

    # prediction of a datapoint category outlier or inlier
    y_pred = clf.predict(X)
    n_inliers = len(y_pred) - np.count_nonzero(y_pred)
    n_outliers = np.count_nonzero(y_pred == 1)

    print("time taken (seconds):", time.time() - start)

    plt.figure(figsize=(10, 10))

    # copy of dataframe
    dfx = df
    dfx['outlier'] = y_pred.tolist()

    # IX1 - inlier feature 1,  IX2 - inlier feature 2
    IX1 = np.array(dfx['Item_MRP'][dfx['outlier'] == 0]).reshape(-1, 1)
    IX2 = np.array(dfx['Item_Outlet_Sales'][dfx['outlier'] == 0]).reshape(-1, 1)

    # OX1 - outlier feature 1, OX2 - outlier feature 2
    OX1 = dfx['Item_MRP'][dfx['outlier'] == 1].values.reshape(-1, 1)
    OX2 = dfx['Item_Outlet_Sales'][dfx['outlier'] == 1].values.reshape(-1, 1)

    print('OUTLIERS:', n_outliers, '|||| INLIERS:', n_inliers, clf_name)

    # threshold value to consider a datapoint inlier or outlier
    threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)

    # decision function calculates the raw anomaly score for every point
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    Z = Z.reshape(xx.shape)

    # fill blue map colormap from minimum anomaly score to threshold value
    #plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.Blues_r)

    # draw red contour line where anomaly score is equal to thresold
    a = plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')

    # fill orange contour lines where range of anomaly score is from threshold to maximum anomaly score
    #plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='orange')

    b = plt.scatter(IX1, IX2, c='white', s=20, edgecolor='k')

    c = plt.scatter(OX1, OX2, c='red', marker="x", s=60, edgecolor='k')

    plt.axis('tight')

    # loc=2 is used for the top left corner
    plt.legend(
        [a.collections[0], b, c],
        ['learned decision function', 'inliers', 'outliers'],
        prop=matplotlib.font_manager.FontProperties(size=20),
        loc=2)

    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title(clf_name)
    plt.show()
