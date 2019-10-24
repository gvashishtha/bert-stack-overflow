# Azure Machine Learning Batch Inference

Azure Machine Learning Batch Inference targets large inference jobs which are not time-sensitive. Batch Inference provides cost-effective inference compute scaling, with unparalleled throughput for asynchronous applications. Batch inference can scale to perform inference on terabytes of production data. Batch Inference is optimized for high throughput, fire-and-forget inference for a large collection of data.

# Get started with batch inference public preview

Batch inference public preview offers mostly stable platform in which to do large inference or generic parallel operations. Below introduces the major steps to use batch inference private preview. For a quick try, please follow prerequisites and simply run the sample notebook provided in this directory.

## Prerequisites

### Python package installation
These packages are not available on pypi yet, and thus require a custom index-url. These commands also assume you are already in a new environment (versioning will clash with pypi packages). If you're unfamiliar with creating a new python environment, you may follow this example for [creating a conda environment](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment#local).
```
# If you have previously installed azureml-* packages, you will need to create a new environment. The versions between pypi and public preview are mutually exclusive.
pip install --extra-index-url https://pypi.python.org/simple --index-url https://azuremlsdktestpypi.azureedge.net/BatchInferencing "azureml-contrib-pipeline-steps<0.1.5" "azureml-widgets<0.1.5"
```
**NOTE**: These packages are not compatible with the versions on pypi (the default pip index). If you want to install additional azureml packages, reuse the --extra-index-url and --index-url arguments from the first installation line.

### Creation of Azure Machine Learning Workspace
If you do not already have a Azure ML Workspace, please run the [configuration Notebook](../../../configuration.ipynb). **NOTE**: Do not install the azureml-sdk package from pypi as described in the configuration notebook, as it's version is not compatible with this private preview.

### Supported Regions
Batch inference private preview supports all production AzureML regions listed [here](https://azure.microsoft.com/en-us/global-infrastructure/services/?regions=all&products=machine-learning-service).

## Configure a batch inference job

To run a batch inference job, you will need to gather some configuration data.

1. **ParallelRunConfig**
    - **entry_script**: scoring script with local file path. If source_directly is present, use relative path, otherwise use any path accessible on machine.
    - **mini_batch_size**: number of records scoring script can process in one inference call (optional, default value is 1)
    - **error_threshold**: percentage of record failures can be ignored and processing should continue. If error percentage goes above this value, job will be aborted. Error threshold is for the entire batch inference job, nor for mini batch. e.g. if you provide an input data folder containing 1000 images and set error threshold to 10 (%), then batch inference will ignore 100 failures but abort job if 101 failures occur.
    - **output_action**: one of the following values
        - **"summary_only"**: scoring script will store the output and batch inference will use the output only for error threshold calculation.
        - **"file"**: for each input file, there will be a corresponding file with the same name in output folder.
        - **"append_row"**: for all input files, only one file will be created in output folder appending all outputs separated by line. File name will be parallel_run_step.txt.
    - **source_directory**: supporting files for scoring (optional)
    - **compute_target**: only **AmlCompute** is supported currently
    - **node_count**: number of compute nodes to be used.
    - **process_count_per_node**: (optional, default value is 1) number of processes per node.
    - **mini_batch_size**: size, in bytes, for chunking dataset into mini-batches (e.g. '1024', '100KB', '1GB').
    - **logging_level**: (optional, default value is 'INFO') log verbosity. Values in increasing verbosity are: 'WARNING', 'INFO', 'DEBUG'.
    - **description**: name given to batch service.
    - **run_invocation_timeout**: run method invocation timeout period in seconds.
    - **environment** (optional)

2. **Scoring (entry) script**: entry point for execution, scoring script should contain two functions
    - **init()**: this function should be used for any costly or common preparation for subsequent inferencing, e.g., loading the model into a global object.
    - **run(input_data, arguments)**: this function will run for each mini-batch.
        - **input_data**: array of locally-accessible file paths.
        - **arguments**: arguments passed during pipeline run will be passed to this function. (See below for example of argument format)
        - **return value**: run() method should return an array. For append_row output_action, these returned elements would be appended into the common output file. For summary_only, the contents of the elements are ignored. For all output actions, each returned output element indicates one successful inference of input element in the input mini-batch.

3. **Base image** (optional)
    - if GPU is required, use DEFAULT_GPU_IMAGE as base image in environment. [Example GPU environment](./file-dataset-image-inference-mnist.ipynb#specify-the-environment-to-run-the-script)

Example image pull:
```python
from azureml.core.runconfig import ContainerRegistry

# use an image available in public Container Registry without authentication
public_base_image = "mcr.microsoft.com/azureml/o16n-sample-user-base/ubuntu-miniconda"

# or use an image available in a private Container Registry
base_image = "myregistry.azurecr.io/mycustomimage:1.0"
base_image_registry = ContainerRegistry()
base_image_registry.address = "myregistry.azurecr.io"
base_image_registry.username = "username"
base_image_registry.password = "password"
```


## Create a batch inference job

**ParallelRunStep** is a newly added step in the azureml.contrib.pipeline.steps package. You will use it to add a step to create a batch inference job with your Azure machine learning pipeline. (Use batch inference without an Azure machine learning pipeline is not supported yet). ParallelRunStep has all the following parameters:
  - **name**: this name will be used to register batch inference service, has the following naming restrictions: (unique, 3-32 chars and regex ^\[a-z\]([-a-z0-9]*[a-z0-9])?$
  - **models**: zero or more model names already registered in Azure Machine Learning model registry.
  - **parallel_run_config**: ParallelRunConfig as defined above.
  - **inputs**: one or more DataSet objects.
  - **output**: this should be a Azure BLOB container path. (other storage platforms are projected to be supported)
  - **arguments**: list of arguments passed to scoring script (optional)
  - **allow_reuse**: optional, default value is True. If the inputs remain the same as a previous run, it will make the previous run results immediately available (skips re-computing the step).

## Passing arguments from pipeline submission to script

Many tasks require arguments to be passed from job submission to the distributed runs. Below is an example to pass such information.
```
# from script which creates pipeline job
parallelrun_step = ParallelRunStep(
  ...
  arguments=["--model_name", "mosaic",     # name of the model we want to use, in case we have more than one option
             "--label_dir", label_dir]     # name of the datastore in which we want all distributed tasks to have a non-sharded instance
)
```
```
# from driver.py/score.py/task.py
import argparse

parser.add_argument('--model_name', dest="model_name")
parser.add_argument('--label_dir', dest="label_dir")

args, unknown_args = parser.parse_known_args()

# to access values
args.model_name # "mosaic"
args.label_dir  # value of 'label_dir' from pipeline-creation script
```

## Submit a batch inference job

You can submit a batch inference job by pipeline_run, or through REST calls with a published pipeline. To control node count using REST API/experiment, please use aml_node_count(special) pipeline parameter. A typical use case follows:

```python
pipeline = Pipeline(workspace=ws, steps=[parallelrun_step])
pipeline_run = Experiment(ws, 'name_of_pipeline_run').submit(pipeline)
```

## Monitor your batch inference job

A batch inference job can take a long time to finish. You can monitor your job's progress from Azure portal, using Azure ML widgets, view console output through SDK, or check out overview.txt in log/azureml directory.

```python
# view with widgets (will display GUI inside a browser)
from azureml.widgets import RunDetails
RunDetails(pipeline_run).show()

# simple console output
pipeline_run.wait_for_completion(show_output=True)
```

# Sample notebooks

-  [file-dataset-image-inference-mnist.ipynb](./file-dataset-image-inference-mnist.ipynb) demonstrates how to use batch inference on an mnist dataset.

# Contacts

  For any questions, please send email to [Azure ML Batch Inference Team](mailto:amlbiteam@microsoft.com).

![Impressions](https://PixelServer20190423114238.azurewebsites.net/api/impressions/MachineLearningNotebooks/contrib/batch_inferencing/README.png) 
