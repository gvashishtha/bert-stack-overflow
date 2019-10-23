from azureml.core import Datastore
from azureml.core import Workspace
from azureml.data.datapath import DataPath, DataPathComputeBinding
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core.runconfig import RunConfiguration, CondaDependencies

aml_workspace=Workspace.get(name='gopalgpu-AML-WS', subscription_id='b17253fa-f327-42d6-9686-f3e553e24763', resource_group='gopalgpu-AML-RG')
datastore_name = 'azureservicedatastore'
existing_datastore = Datastore.get(aml_workspace, datastore_name)
training_dataset = DataPath(datastore=existing_datastore, path_on_datastore='azure-service-data/')
data_dir = (PipelineParameter(name="data_dir", default_value=training_dataset), DataPathComputeBinding())
data_dir = PipelineParameter(
    name="data_dir", default_value='https://raw.githubusercontent.com/gvashishtha/tfworld_data/master/') #default_value=datastore.path('azure-service-data/').as_mount())
max_seq_length = PipelineParameter(
    name="max_seq_length", default_value=128)
learning_rate = PipelineParameter(
    name="learning_rate", default_value=3e-5)
num_epochs = PipelineParameter(
    name="num_epochs", default_value=5)
export_dir = PipelineParameter(
    name="export_dir", default_value="./outputs/exports")
batch_size = PipelineParameter(
    name="batch_size", default_value=32)
steps_per_epoch = PipelineParameter(
    name="steps_per_epoch", default_value=150)
run_config = RunConfiguration(conda_dependencies=CondaDependencies.create(
    conda_packages=['numpy', 'pandas',
                    'scikit-learn', 'keras'],
    pip_packages=['azure', 'azureml-sdk',
                    'azure-storage',
                    'azure-storage-blob',
                    'transformers>=2.1.1',
                    'tensorflow>=2.0.0',
                    'tensorflow-gpu>=2.0.0'])
)
run_config.environment.docker.enabled = True
train_step = PythonScriptStep(
    name="Train Model",
    script_name='training/train.py',
    compute_target=aml_workspace.compute_targets['train-cluster'],
    source_directory='code',
    # inputs=[data_dir],
    arguments=[
        "--data_dir", data_dir,
        "--max_seq_length", max_seq_length,
        "--learning_rate", learning_rate,
        "--num_epochs", num_epochs,
        "--export_dir", export_dir,
        "--batch_size", batch_size, 
        "--steps_per_epoch", steps_per_epoch
    ],
    runconfig=run_config,
    allow_reuse=False,
)
steps = [train_step]
train_pipeline = Pipeline(workspace=aml_workspace, steps=steps)
train_pipeline.validate()
published_pipeline = train_pipeline.publish(
    name='test_git_data',
    description="Git data instead of datastore",
    version=1
)
# pipeline_parameters = {"data_dir": DataPath(datastore=existing_datastore, path_on_datastore="azure-service-data/"), "max_seq_length": 128, "learning_rate": 3e-5, 
#                            "num_epochs": 5, "export_dir": "./outputs/exports"}
response = published_pipeline.submit(
               workspace=aml_workspace,
               experiment_name='test_experiment')

from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import LocalWebservice
model=aml_workspace.models['tf_model.h5']
model.download(target_dir='./from_azure', exist_ok=False, exists_ok=None)
inference_config = InferenceConfig(runtime="python",
                                   entry_script="./code/scoring/score.py",
                                   conda_file="./code/scoring/conda_dependencies.yml")
deployment_config = LocalWebservice.deploy_configuration(port=8890)

from transformers import TFBertPreTrainedModel, TFBertMainLayer, BertTokenizer
import tensorflow as tf
class TFBertForMultiClassification(TFBertPreTrainedModel):
    def __init__(self, config, *inputs, **kwargs):
        super(TFBertForMultiClassification, self).__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.bert = TFBertMainLayer(config, name='bert')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(config.num_labels,
                                                kernel_initializer=get_initializer(config.initializer_range),
                                                name='classifier',
                                                activation='softmax')
    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=kwargs.get('training', False))
        logits = self.classifier(pooled_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        return outputs  # logits, (hidden_states), (attentions)

max_seq_length = 128
labels = ['azure-web-app-service', 'azure-storage', 'azure-devops', 'azure-virtual-machine', 'azure-functions']


# Deploy the service
local_service = Model.deploy(
   aml_workspace, "mymodel", [model], inference_config, deployment_config)
# Wait for the deployment to complete
local_service.wait_for_deployment(True)
# Display the port that the web service is available on
print(local_service.port)


## 

import json
import requests
url = 'http://52.151.6.0:80/api/v1/service/mlopspython-aks/score'
api_key= 'lx8Bj1rNMDLtg9hWd1c2KvLIOfP8oHe8'
payload = {'text': 'I am trying to release a website'}
headers = {'content-type': 'application/json', 'Authorization':('Bearer '+ api_key)}
response = requests.post(url, data=json.dumps(payload), headers=headers)
