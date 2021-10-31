import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVR
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=40)

# Average CV score on the training set was: -0.10706350379252258
exported_pipeline = make_pipeline(
    MaxAbsScaler(),
    LinearSVR(C=0.1, dual=True, epsilon=0.001, loss="epsilon_insensitive", tol=0.0001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 40)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
