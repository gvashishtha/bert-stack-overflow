from azureml.core import Workspace
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.exceptions import ComputeTargetException


def get_aks(
    workspace: Workspace,
    compute_name: str
):
    # Verify that cluster does not exist already
    try:
        aks_target = workspace.compute_targets.get(compute_name)
        if aks_target is not None and type(aks_target) is AksCompute:
            print('Found existing compute target ' + compute_name
                + ' so using it.')  # noqa: E127
        else:
            prov_config = AksCompute.provisioning_configuration(
                cluster_purpose=AksCompute.ClusterPurpose.DEV_TEST)
            aks_name = compute_name

            print("No Azure Kubernetes Service cluster found, "
                  "creating one now...")

            # Create the cluster
            aks_target = ComputeTarget.create(
                workspace=workspace,
                name=aks_name,
                provisioning_configuration=prov_config)

            # Wait for the create process to complete
            aks_target.wait_for_completion(show_output=True)
        return aks_target
    except ComputeTargetException as e:
        print(e)
        print('An error occurred trying to provision compute.')
        raise
