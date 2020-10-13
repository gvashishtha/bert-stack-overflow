# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import os
import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from azureml.core.run import Run

OUTPUT_DIR = './outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# get the IBM employee attrition dataset
outdirname = 'dataset.6.21.19'
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
zipfilename = outdirname + '.zip'
urlretrieve('https://publictestdatasets.blob.core.windows.net/data/' + zipfilename, zipfilename)
with zipfile.ZipFile(zipfilename, 'r') as unzip:
    unzip.extractall('.')
attritionData = pd.read_csv('./WA_Fn-UseC_-HR-Employee-Attrition.csv')

# dropping Employee count as all values are 1 and hence attrition is independent of this feature
attritionData = attritionData.drop(['EmployeeCount'], axis=1)
# dropping Employee Number since it is merely an identifier
attritionData = attritionData.drop(['EmployeeNumber'], axis=1)
attritionData = attritionData.drop(['Over18'], axis=1)
# since all values are 80
attritionData = attritionData.drop(['StandardHours'], axis=1)

# converting target variables from string to numerical values
target_map = {'Yes': 1, 'No': 0}
attritionData["Attrition_numerical"] = attritionData["Attrition"].apply(lambda x: target_map[x])
target = attritionData["Attrition_numerical"]

attritionXData = attritionData.drop(['Attrition_numerical', 'Attrition'], axis=1)

# Creating dummy columns for each categorical feature
categorical = []
for col, value in attritionXData.iteritems():
    if value.dtype == 'object':
        categorical.append(col)

# Store the numerical columns in a list numerical
numerical = attritionXData.columns.difference(categorical)  


from sklearn.compose import ColumnTransformer

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocess = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical),
        ('cat', categorical_transformer, categorical)])

pipeline = make_pipeline(preprocess)


X_train, X_test, y_train, y_test = train_test_split(attritionXData, 
                                                    target, 
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=target)

X_train_t = pipeline.fit_transform(X_train)
X_test_t = pipeline.transform(X_test)

# check tensorflow version
import tensorflow as tf
from distutils.version import StrictVersion

print(tf.__version__)
# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.


network = tf.keras.models.Sequential()
network.add(tf.keras.layers.Dense(units=16, activation='relu', input_shape=(X_train_t.shape[1],)))
network.add(tf.keras.layers.Dense(units=16, activation='relu'))
network.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compile neural network
network.compile(loss='binary_crossentropy', # Cross-entropy
                optimizer='rmsprop', # Root Mean Square Propagation
                metrics=['accuracy']) # Accuracy performance metric

# Train neural network
history = network.fit(X_train_t, # Features
                      y_train, # Target vector
                      epochs=20, # Number of epochs
                      verbose=1, # Print description after each epoch
                      batch_size=100, # Number of observations per batch
                      validation_data=(X_test_t, y_test)) # Data for evaluation




# You can run the DeepExplainer directly, or run the TabularExplainer which will choose the most appropriate explainer
from interpret.ext.greybox import DeepExplainer
explainer = DeepExplainer(network,
                             X_train,
                             features=X_train.columns,
                             classes=["STAYING", "LEAVING"], 
                             transformations=preprocess,
                             model_task="classification",
                             is_classifier=True
                            )

# you can use the training data or the test data here
global_explanation = explainer.explain_global(X_test)
# You can pass a specific data point or a group of data points to the explain_local function
# E.g., Explain the first data point in the test set
instance_num = 1
local_explanation = explainer.explain_local(X_test[:instance_num])


# get the run this was submitted from to interact with run history
run = Run.get_context()

from azureml.contrib.interpret.explanation.explanation_client import ExplanationClient

# create an explanation client to store the explanation (contrib API)
client = ExplanationClient.from_run(run)

# uploading model explanation data for storage or visualization
comment = 'Global explanation on classification model trained on IBM employee attrition dataset'
client.upload_model_explanation(global_explanation, comment=comment)

network.save('./outputs/model.h5')
