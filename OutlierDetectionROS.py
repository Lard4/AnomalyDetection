#!/usr/bin/env python
# ROS import
import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from std_msgs.msg import Int32
from dynamic_reconfigure.server import Server
from fault_tolerance.cfg import OutlierDetectionROSConfig

# Other Python Imports
import time
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.font_manager
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# PyOD - Import models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.utils.data import generate_data, get_outliers_inliers

## Dictionaries for publishers and subscribers
publishers = {}
subscribers = {}

# Outlier detection parameters
random_state = np.random.RandomState(42)
outliers_fraction = 0.05
MAX_OUTLIERS_BETWEEN_CHECKS = 10
SECS_BETWEEN_OUTLIER_CHECKS = 1
SUB_CALLS_BETWEEN_OD_RUN = 50
RUN_VALID = False
outliers_since_last_check = 0
# Define outlier detection classifiers
classifiers = {
        'Angle-based Outlier Detector (ABOD_5)': {'enabled' : False, 'classifier' : ABOD(contamination=outliers_fraction, n_neighbors=5)},
        'Angle-based Outlier Detector (ABOD_25)': {'enabled' : False, 'classifier' : ABOD(contamination=outliers_fraction, n_neighbors=25)},
        'Angle-based Outlier Detector (ABOD_50)': {'enabled' : False, 'classifier' : ABOD(contamination=outliers_fraction, n_neighbors=50)},
        'Angle-based Outlier Detector (ABOD_100)': {'enabled' : False, 'classifier' : ABOD(contamination=outliers_fraction, n_neighbors=100)},
        'Angle-based Outlier Detector (ABOD_500)': {'enabled' : False, 'classifier' : ABOD(contamination=outliers_fraction, n_neighbors=500)},
        'Angle-based Outlier Detector (ABOD_8000)': {'enabled' : False, 'classifier' : ABOD(contamination=outliers_fraction, n_neighbors=8000)},
        'Cluster-based Local Outlier Factor (CBLOF)': {'enabled' : False, 'classifier' : CBLOF(contamination=outliers_fraction,check_estimator=False, random_state=random_state)},
        'Feature Bagging': {'enabled' : False, 'classifier' : FeatureBagging(LOF(n_neighbors=35),contamination=outliers_fraction,check_estimator=False,random_state=random_state,)},
        'Histogram-base Outlier Detection (HBOS)': {'enabled' : False, 'classifier' : HBOS(contamination=outliers_fraction)},
        'Isolation Forest': {'enabled' : False, 'classifier' : IForest(contamination=outliers_fraction,random_state=random_state, behaviour="new")},
        'K Nearest Neighbors (KNN_5)': {'enabled' : True, 'classifier' : KNN(contamination=outliers_fraction, n_neighbors=5)},
        'K Nearest Neighbors (KNN_25)': {'enabled' : False, 'classifier' : KNN(contamination=outliers_fraction, n_neighbors=25)},
        'K Nearest Neighbors (KNN_50)': {'enabled' : False, 'classifier' : KNN(contamination=outliers_fraction, n_neighbors=50)},
        'K Nearest Neighbors (KNN_100)': {'enabled' : False, 'classifier' : KNN(contamination=outliers_fraction, n_neighbors=100)},
        'K Nearest Neighbors (KNN_500)': {'enabled' : False, 'classifier' : KNN(contamination=outliers_fraction, n_neighbors=500)},
        'K Nearest Neighbors (KNN_8000)': {'enabled' : False, 'classifier' : KNN(contamination=outliers_fraction, n_neighbors=8000)},
        'Average KNN': {'enabled' : False, 'classifier' : KNN(method='mean',contamination=outliers_fraction)}
}

# List variables
MAX_INDEX = 250
ARRAYS_FILLED_ONCE = False 
index = 0
# Lists to hold imu data
imu_ang_x_data = [0] * MAX_INDEX
imu_ang_y_data = [0] * MAX_INDEX
imu_ang_z_data = [0] * MAX_INDEX

def serverCallback(config, level):
    """ dynamic_reconfigure server callback function.
        
        See cfg/OutlierDetectionROS.cfg for defined configurables
    """
    global classifiers, outliers_fraction, MAX_INDEX, MAX_OUTLIERS_BETWEEN_CHECKS, \
            SECS_BETWEEN_OUTLIER_CHECKS, SUB_CALLS_BETWEEN_OD_RUN

    rospy.loginfo("""OD: Reconfigure request received""")

    enabled = bool(config['override_enable'])
    if not enabled:
        return config

    # Etc reconfigurables
    outliers_fraction = float(config['outliers_fraction'])
    #MAX_INDEX = int(config['max_index'])
    MAX_OUTLIERS_BETWEEN_CHECKS = int(config['max_outliers_between_checks'])
    SECS_BETWEEN_OUTLIER_CHECKS = float(config['secs_between_outlier_checks'])
    SUB_CALLS_BETWEEN_OD_RUN = int(config['sub_calls_between_od_run'])

    # Classifiers
    classifiers['Angle-based Outlier Detector (ABOD_5)']['enabled'] = bool(config['ABOD_5'])
    classifiers['Cluster-based Local Outlier Factor (CBLOF)']['enabled'] = bool(config['CBLOF'])
    classifiers['Feature Bagging']['enabled'] = bool(config['FB'])
    classifiers['Histogram-base Outlier Detection (HBOS)']['enabled'] = bool(config['HBOS'])
    classifiers['Isolation Forest']['enabled'] = bool(config['IF'])
    classifiers['K Nearest Neighbors (KNN_5)']['enabled'] = bool(config['KNN_5'])
    classifiers['Average KNN']['enabled'] = bool(config['A_KNN'])

    rospy.loginfo("""here1""")

    return config

def run_outlier_detection():
    global imu_ang_x_data, imu_ang_y_data, imu_ang_z_data, \
            outliers_since_last_check, RUN_VALID
    imu_data = [imu_ang_x_data, imu_ang_y_data, imu_ang_z_data]
    X = np.transpose(np.array(imu_data))
    #print(X)

    for i, (clf_name, clf_properties) in enumerate(classifiers.items()):
        if clf_properties['enabled'] == False:
            continue
        clf = clf_properties['classifier']
        start = time.time()
        clf.fit(X)
        # predict raw anomaly score
        scores_pred = clf.decision_function(X) * -1

        # prediction of a datapoint category outlier or inlier
        y_pred = clf.predict(X)
        n_inliers = len(y_pred) - np.count_nonzero(y_pred)
        n_outliers = np.count_nonzero(y_pred == 1)

        outliers_since_last_check += n_outliers
        publishers['imu_outliers']['publisher'].publish(outliers_since_last_check)

        rospy.loginfo('OUTLIERS: {} |||| INLIERS: {} with strategy:  {}'.format(n_outliers, n_inliers, clf_name))
        rospy.loginfo("Time taken (seconds): {}".format(time.time() - start))

        #sub_calls = subscribers['imu']['calls']
        #print("Total subscriber calls: {}".format(sub_calls))
    RUN_VALID = False

def check_outliers():
    global outliers_since_last_check
    status = ""
    if(outliers_since_last_check > MAX_OUTLIERS_BETWEEN_CHECKS):
        status = "FAULT DETECTED"
    else:
        status = "Good"
    publishers['imu_outlier_status']['publisher'].publish(status)

def update_imu_data_callback(data):
    global imu_ang_x_data, imu_ang_y_data, imu_ang_z_data, subscribers, index, ARRAYS_FILLED_ONCE, RUN_VALID
    imu_ang_x_data[index] = data.angular_velocity.x
    imu_ang_y_data[index] = data.angular_velocity.y
    imu_ang_z_data[index] = data.angular_velocity.z
    index = (index + 1) % MAX_INDEX
    subscribers['imu']['calls'] += 1

    if(index % SUB_CALLS_BETWEEN_OD_RUN == 0):
        RUN_VALID = True

    if((ARRAYS_FILLED_ONCE == False) and (index == MAX_INDEX - 1)):
        ARRAYS_FILLED_ONCE = True

def initializeSubscribers():
    """ Initialize ROS subscribers. """
    global subscribers
    subscribers['imu'] = {}
    subscribers['imu']['subscriber'] = rospy.Subscriber("imu", Imu, update_imu_data_callback)
    subscribers['imu']['calls'] = 0

def initializePublishers():
    """ Initialize ROS publishers. """
    global publishers
    publishers['imu_outlier_status'] = {}
    publishers['imu_outlier_status']['publisher'] = rospy.Publisher('OD/imu_outlier_status', String, queue_size=1)
    publishers['imu_outlier_status']['calls'] = 0

    publishers['imu_outliers'] = {}
    publishers['imu_outliers']['publisher'] = rospy.Publisher('OD/imu_outliers', Int32, queue_size=1)
    publishers['imu_outliers']['calls'] = 0

    publishers['rate'] = {}
    publishers['rate']['publisher'] = rospy.Publisher('OD/rate', Int32, queue_size=1)
    publishers['rate']['calls'] = 0

def main():
    """ Main program. """
    global outliers_since_last_check

    initializePublishers()
    initializeSubscribers()

    rospy.init_node('OutlierDetectionROS')
    # Initialize dynamic_reconfigure
    server = Server(OutlierDetectionROSConfig, serverCallback)
    rate = rospy.Rate(120)
    seq = 0
    
    curr_time = time.time()
    prev_time = curr_time
    while not rospy.is_shutdown():
        curr_time = time.time()
        #rospy.loginfo("Seq number: {}".format(seq))
        seq += 1

        if(ARRAYS_FILLED_ONCE and RUN_VALID):
            run_outlier_detection()

        if(curr_time - prev_time > SECS_BETWEEN_OUTLIER_CHECKS):
            check_outliers()
            outliers_since_last_check = 0
            prev_time = curr_time

        publishers['rate']['publisher'].publish(seq)
        rate.sleep()

if __name__ == '__main__':
    rospy.loginfo("Starting OutlierDetectionROS node...")
    try:
        main()
    except rospy.ROSInterruptException:
        pass

