import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, MaxAbsScaler
from sklearn.svm import LinearSVR
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=40)

# Average CV score on the training set was: -0.1063395350637311
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=18),
    MaxAbsScaler(),
    VarianceThreshold(threshold=0.001),
    Binarizer(threshold=0.9),
    LinearSVR(C=10.0, dual=True, epsilon=0.0001, loss="epsilon_insensitive", tol=0.0001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 40)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
