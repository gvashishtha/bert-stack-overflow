import sys
import os
import azureml

sys.path.insert(1,   # NOQA: E402
    os.path.abspath("./3-ML-Ops/util"))   # NOQA: E402
from distutils.version import StrictVersion
from workspace import get_workspace


# Just an example of a unit test against
# a utility function common_scoring.next_saturday
def test_get_workspace():
    workspace_name = os.environ.get("BASE_NAME")+"-AML-WS"
    resource_group = "AML-RG-"+os.environ.get("BASE_NAME")
    subscription_id = os.environ.get("SUBSCRIPTION_ID")
    tenant_id = os.environ.get("TENANT_ID")
    app_id = os.environ.get("SP_APP_ID")
    app_secret = os.environ.get("SP_APP_SECRET")

    aml_workspace = get_workspace(
        workspace_name,
        resource_group,
        subscription_id,
        tenant_id,
        app_id,
        app_secret)

    assert aml_workspace.name == workspace_name


'''
test_versions

Checks that the azureml SDK version and tensorflow version are what we expect

'''


def test_versions():
    # assert(StrictVersion(tf.__version__) >= StrictVersion('2.0'))
    assert(StrictVersion(azureml.core.__version__) >= StrictVersion('1.0.65'))
