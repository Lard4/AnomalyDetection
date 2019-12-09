###
### Original source: https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/
###
### Modified and redisgned to fit an anomaly detection ROS topic for a
### fault tolerant robot for CPE 446 at Cal Poly SLO.
###
### Authors: Lakshay Arora, Kevin Dixson, Sukhman Marok
### 

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # <-- Note the capitalization!
from pyod.models.abod import ABOD
from pyod.models.iforest import IForest
from pyod.models.knn import KNN

df = pd.read_csv("imu_injected_data.csv")
df.plot.scatter('field.angular_velocity.x', 'field.angular_velocity.z')

from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler(feature_range=(0, 1))
#df[['field.angular_velocity.x', 'field.angular_velocity.y', 'field.angular_velocity.z']] = \
#    scaler.fit_transform(df[['field.angular_velocity.x', 'field.angular_velocity.y', 'field.angular_velocity.z']])
#df[['field.angular_velocity.x', 'field.angular_velocity.y', 'field.angular_velocity.z']].head()
X1 = df['field.angular_velocity.x'].values.reshape(-1, 1)
X2 = df['field.angular_velocity.z'].values.reshape(-1, 1)
X3 = df['field.angular_velocity.y'].values.reshape(-1, 1)

X = np.concatenate([X1, X2, X3], axis=1)

print(X)

random_state = np.random.RandomState(42)
outliers_fraction = 0.1
classifiers = {
    'Angle-based Outlier Detector (ABOD-5)': ABOD(contamination=outliers_fraction, n_neighbors=5),
    'Isolation Forest': IForest(contamination=outliers_fraction, random_state=random_state, behaviour="new"),
    'K Nearest Neighbors (KNN-5)': KNN(contamination=outliers_fraction, n_neighbors=5),
    'K Nearest Neighbors (KNN-25)': KNN(contamination=outliers_fraction, n_neighbors=25),
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
    print('OUTLIERS:', n_outliers, '|||| INLIERS:', n_inliers, clf_name)
    print("\n\n\n\n")

    fig = plt.figure()

    # copy of dataframe
    dfx = df
    dfx['outlier'] = y_pred.tolist()

    # IX1 - inlier feature 1,  IX2 - inlier feature 2
    IX1 = np.array(dfx['field.angular_velocity.x'][dfx['outlier'] == 0]).reshape(-1, 1)
    IX2 = np.array(dfx['field.angular_velocity.y'][dfx['outlier'] == 0]).reshape(-1, 1)
    IX3 = np.array(dfx['field.angular_velocity.z'][dfx['outlier'] == 0]).reshape(-1, 1)

    # OX1 - outlier feature 1, OX2 - outlier feature 2
    OX1 = dfx['field.angular_velocity.x'][dfx['outlier'] == 1].values.reshape(-1, 1)
    OX2 = dfx['field.angular_velocity.y'][dfx['outlier'] == 1].values.reshape(-1, 1)
    OX3 = dfx['field.angular_velocity.z'][dfx['outlier'] == 1].values.reshape(-1, 1)

    ax = Axes3D(fig)
    iset = ax.scatter(IX1, IX2, IX3, c="black", label="inliers")
    oset = ax.scatter(OX1, OX2, OX3, c="red", s=60, marker="x", label="outliers")
    ax.legend()
    ax.set_xlabel("angular_velocity.x")
    ax.set_ylabel("angular_velocity.y")
    ax.set_zlabel("angular_velocity.z")
    ax.set_title(clf_name)
    for angle in range(0, 360, 10):
        ax.view_init(30, angle)
        fig.savefig(clf_name + str(angle) + '.png')  # save the figure to file
        plt.close(fig)
