from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import EstimatorStep, PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core.runconfig import RunConfiguration, CondaDependencies
from azureml.core import Dataset, Datastore
from azureml.train.dnn import TensorFlow
import os
import sys
from dotenv import load_dotenv
sys.path.insert(1, os.path.abspath("./3-ML-Ops/util"))  # NOQA: E402
from workspace import get_workspace
from attach_compute import get_compute
from attach_aks import get_aks


def main():
    load_dotenv()
    workspace_name = os.environ.get("BASE_NAME")+"-AML-WS"
    resource_group = "AML-RG-"+os.environ.get("BASE_NAME")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    tenant_id = os.environ.get("TENANT_ID")
    app_id = os.environ.get("SP_APP_ID")
    app_secret = os.environ.get("SP_APP_SECRET")
    sources_directory_train = os.environ.get("SOURCES_DIR_TRAIN")
    train_script_path = os.environ.get("TRAIN_SCRIPT_PATH")
    evaluate_script_path = os.environ.get("EVALUATE_SCRIPT_PATH")
    vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU")
    compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME")
    aks_name = os.environ.get("AKS_CLUSTER_NAME")
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

    datastore_name = 'tfworld'
    container_name = 'azureml-blobstore-7c6bdd88-21fa-453a-9c80-16998f02935f'
    account_name = 'tfworld6818510241'
    sas_token = '?sv=2019-02-02&ss=bfqt&srt=sco&sp=rl&se=2019-11-08T05:12:15Z&st=2019-10-23T20:12:15Z&spr=https&sig=eDqnc51TkqiIklpQfloT5vcU70pgzDuKb5PAGTvCdx4%3D'  # noqa: E501

    try:
        existing_datastore = Datastore.get(aml_workspace, datastore_name)
    except:  # noqa: E722
        existing_datastore = Datastore \
            .register_azure_blob_container(workspace=aml_workspace,
                                           datastore_name=datastore_name,
                                           container_name=container_name,
                                           account_name=account_name,
                                           sas_token=sas_token
                                           )

    azure_dataset = Dataset.File.from_files(
        path=(existing_datastore, 'azure-service-classifier/data'))
    azure_dataset = azure_dataset.register(
        workspace=aml_workspace,
        name='Azure Services Dataset',
        description='Dataset containing azure related posts on Stackoverflow',
        create_new_version=True)

    azure_dataset.to_path()
    input_data = azure_dataset.as_named_input('input_data1').as_mount(
        '/tmp/data')

    model_name = PipelineParameter(
        name="model_name", default_value=model_name)
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

    # initialize the TensorFlow estimator
    estimator = TensorFlow(
        source_directory=sources_directory_train,
        entry_script=train_script_path,
        compute_target=aml_compute,
        framework_version='2.0',
        use_gpu=True,
        pip_packages=[
            'transformers==2.0.0',
            'azureml-dataprep[fuse,pandas]==1.1.22'])

    train_step = EstimatorStep(
        name="Train Model",
        estimator=estimator,
        estimator_entry_script_arguments=[
            "--data_dir", input_data,
            "--max_seq_length", max_seq_length,
            "--learning_rate", learning_rate,
            "--num_epochs", num_epochs,
            "--export_dir", export_dir,
            "--batch_size", batch_size,
            "--steps_per_epoch", steps_per_epoch],
        compute_target=aml_compute,
        inputs=[input_data],
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
            "--build_id", build_id,
        ],
        runconfig=run_config,
        allow_reuse=False,
    )
    print("Step Evaluate created")

    # Currently, the Evaluate step will automatically register
    # the model if it performs better. This step is based on a
    # previous version of the repo which utilized JSON files to
    # track evaluation results.

    evaluate_step.run_after(train_step)
    steps = [evaluate_step]

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

    # Get AKS cluster for deployment
    aks_compute = get_aks(
        aml_workspace,
        aks_name
    )
    if aks_compute is not None:
        print(aks_compute)


if __name__ == '__main__':
    main()
