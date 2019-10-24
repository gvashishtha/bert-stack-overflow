from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline  # , PipelineData
from azureml.core.runconfig import RunConfiguration, CondaDependencies
from azureml.core import Datastore
from azureml.data.datapath import DataPath, DataPathComputeBinding
import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.abspath("./ml_service/util"))  # NOQA: E402
from workspace import get_workspace
from attach_compute import get_compute
from msrest.exceptions import HttpOperationError


def main():
    load_dotenv()
    workspace_name = os.environ.get("BASE_NAME")+"-AML-WS"
    resource_group = os.environ.get("BASE_NAME")+"-AML-RG"
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    tenant_id = os.environ.get("TENANT_ID")
    app_id = os.environ.get("SP_APP_ID")
    app_secret = os.environ.get("SP_APP_SECRET")
    sources_directory_train = os.environ.get("SOURCES_DIR_TRAIN")
    train_script_path = os.environ.get("TRAIN_SCRIPT_PATH")
    evaluate_script_path = os.environ.get("EVALUATE_SCRIPT_PATH")
    # register_script_path = os.environ.get("REGISTER_SCRIPT_PATH")
    vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU")
    compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME")
    model_name = os.environ.get("MODEL_NAME")
    build_id = os.environ.get("BUILD_BUILDID")
    pipeline_name = os.environ.get("TRAINING_PIPELINE_NAME")
    experiment_name = os.environ.get("EXPERIMENT_NAME")

    # Get Azure machine learning workspace
    aml_workspace = get_workspace(
        workspace_name,
        resource_group,
        subscription_id,
        tenant_id,
        app_id,
        app_secret)
    print(aml_workspace)

    # Get Azure machine learning cluster
    aml_compute = get_compute(
        aml_workspace,
        compute_name,
        vm_size)
    if aml_compute is not None:
        print(aml_compute)

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

    datastore_name = 'azureservicedatastore'
    container_name = 'azureml-blobstore-4eb52bd9-5943-4781-822b-d70b34687789'
    account_name = 'johneast6815302561'
    account_key = os.environ.get('DS_KEY')
    try:
        existing_datastore = Datastore.get(aml_workspace, datastore_name)
    except HttpOperationError:
        existing_datastore = Datastore \
            .register_azure_blob_container(workspace=aml_workspace,
                                           datastore_name=datastore_name,
                                           container_name=container_name,
                                           account_name=account_name,
                                           account_key=account_key
                                           )

    training_dataset = DataPath(datastore=existing_datastore,
                                path_on_datastore='azure-service-data/'
                                )

# TODO: figure out why we bother with default values
    model_name = PipelineParameter(
        name="model_name", default_value=model_name)
    # release_id = PipelineParameter(
    #     name="release_id", default_value="0")
    data_dir = (PipelineParameter(name="data_dir",
                                  default_value=training_dataset),
                DataPathComputeBinding()
                )
    max_seq_length = PipelineParameter(
        name="max_seq_length", default_value=128)
    learning_rate = PipelineParameter(
        name="learning_rate", default_value=3e-5)
    num_epochs = PipelineParameter(
        name="num_epochs", default_value=3)
    export_dir = PipelineParameter(
        name="export_dir", default_value="./outputs/exports")
    batch_size = PipelineParameter(
        name="batch_size", default_value=32)
    steps_per_epoch = PipelineParameter(
        name="steps_per_epoch", default_value=100)

    train_step = PythonScriptStep(
        name="Train Model",
        script_name=train_script_path,
        compute_target=aml_compute,
        source_directory=sources_directory_train,
        inputs=[data_dir],
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
    print("Step Train created")

    evaluate_step = PythonScriptStep(
        name="Evaluate Model ",
        script_name=evaluate_script_path,
        compute_target=aml_compute,
        source_directory=sources_directory_train,
        arguments=[
            "--model_name", model_name,
        ],
        runconfig=run_config,
        allow_reuse=False,
    )
    print("Step Evaluate created")

    # Currently, the Evaluate step will automatically register
    # the model if it performs better. This step is based on a
    # previous version of the repo which utilized JSON files to
    # track evaluation results.

    # register_model_step = PythonScriptStep(
    #     name="Register New Trained Model",
    #     script_name=register_script_path,
    #     compute_target=aml_compute,
    #     source_directory=sources_directory_train,
    #     arguments=[
    #         "--release_id", release_id,
    #         "--model_name", model_name,
    #     ],
    #     runconfig=run_config,
    #     allow_reuse=False,
    # )
    # print("Step register model created")

    evaluate_step.run_after(train_step)
    # register_model_step.run_after(evaluate_step)
    steps = [evaluate_step]  # [train_step, evaluate_step]

    train_pipeline = Pipeline(workspace=aml_workspace, steps=steps)
    train_pipeline.validate()
    published_pipeline = train_pipeline.publish(
        name=pipeline_name,
        description="Model training/retraining pipeline",
        version=build_id
    )
    print(f'Published pipeline: {published_pipeline.name}')
    print(f'for build {published_pipeline.version}')

    response = published_pipeline.submit(  # noqa: F841
               workspace=aml_workspace,
               experiment_name=experiment_name)


if __name__ == '__main__':
    main()
