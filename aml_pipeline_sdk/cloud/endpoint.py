"""Creates and invokes a managed online endpoint."""

import logging
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.entities import ManagedOnlineDeployment, ManagedOnlineEndpoint
from azure.identity import DefaultAzureCredential

from common import MODEL_NAME

ENDPOINT_NAME = "endpoint-pipeline-sdk"
DEPLOYMENT_NAME = "blue"
TEST_DATA_PATH = Path(
    Path(__file__).parent.parent, "test_data", "images_azureml.json")


def main():
    logging.basicConfig(level=logging.INFO)
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)

    # Create the managed online endpoint.
    logging.info("Creating the managed online endpoint...")
    endpoint = ManagedOnlineEndpoint(
        name=ENDPOINT_NAME,
        auth_mode="key",
    )
    registered_endpoint = ml_client.online_endpoints.begin_create_or_update(
        endpoint)
    if registered_endpoint is not None:
        logging.info("Managed online endpoint with name `%s created.",
                     registered_endpoint.name)

    # Get the latest version of the registered model.
    latest_model_version = max(
        [int(m.version) for m in ml_client.models.list(name=MODEL_NAME)])
    registered_model = ml_client.models.get(name=MODEL_NAME,
                                            version=latest_model_version)
    logging.info(
        "Deployment will be created with model named `%s` with version %s.",
        registered_model.name, registered_model.version)

    # Create the managed online deployment.
    logging.info("Creating the managed online deployment...")
    deployment = ManagedOnlineDeployment(name=DEPLOYMENT_NAME,
                                         endpoint_name=ENDPOINT_NAME,
                                         model=registered_model,
                                         instance_type="Standard_DS4_v2",
                                         instance_count=1)
    registered_deployment = ml_client.online_deployments.begin_create_or_update(
        deployment)
    if registered_deployment is not None:
        logging.info("Managed online deployment with name `%s created.",
                     registered_deployment.name)

    # Set deployment traffic to 100%.
    registered_endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(registered_endpoint)

    # Invoke the endpoint.
    logging.info("Invoking the endpoint...")
    result = ml_client.online_endpoints.invoke(endpoint_name=ENDPOINT_NAME,
                                               deployment_name=DEPLOYMENT_NAME,
                                               request_file=TEST_DATA_PATH)
    logging.info(result)


if __name__ == "__main__":
    main()
