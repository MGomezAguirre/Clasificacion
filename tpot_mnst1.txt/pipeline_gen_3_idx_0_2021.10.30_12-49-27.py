import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -0.10871310637822239
exported_pipeline = LinearSVR(C=0.01, dual=True, epsilon=0.0001, loss="epsilon_insensitive", tol=0.0001)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
