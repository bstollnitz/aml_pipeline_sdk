"""Creates and runs an Azure ML pipeline."""

import logging
from pathlib import Path
from typing import Dict

from azure.ai.ml import Input, MLClient, load_component
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import AmlCompute, Data, Model
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential

COMPUTE_NAME = "cluster-gpu"
DATA_NAME = "data-fashion-mnist"
DATA_VERSION = "1"
EXPERIMENT_NAME = "aml-pipeline-sdk"
DATA_PATH = Path(Path(__file__).parent.parent, "data")
TRAIN_PATH = Path(Path(__file__).parent, "train.yml")
TEST_PATH = Path(Path(__file__).parent, "test.yml")
MODEL_NAME = "model-pipeline-sdk"
MODEL_VERSION = "1"


def main():
    logging.basicConfig(level=logging.INFO)
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)

    # Create the compute cluster.
    cluster_gpu = AmlCompute(
        name=COMPUTE_NAME,
        type="amlcompute",
        size="STANDARD_NC6S_V3",
        location="westus",
        min_instances=0,
        max_instances=4,
    )
    ml_client.begin_create_or_update(cluster_gpu)

    # Create the data set.
    try:
        registered_dataset = ml_client.data.get(name=DATA_NAME,
                                                version=DATA_VERSION)
    except ResourceNotFoundError:
        dataset = Data(
            path=DATA_PATH,
            type=AssetTypes.URI_FOLDER,
            description="Fashion MNIST data set",
            name=DATA_NAME,
            version=DATA_VERSION,
        )
        ml_client.data.create_or_update(dataset)
        registered_dataset = ml_client.data.get(name=DATA_NAME,
                                                version=DATA_VERSION)

    # Create the components.
    train_component = load_component(path=TRAIN_PATH)
    test_component = load_component(path=TEST_PATH)

    # TODO: Can we create the components without using YAML files?

    try:
        registered_train_component = ml_client.components.get(
            name=train_component.name, version=train_component.version)
    except ResourceNotFoundError:
        registered_train_component = ml_client.components.create_or_update(
            train_component)

    try:
        registered_test_component = ml_client.components.get(
            name=test_component.name, version=test_component.version)
    except ResourceNotFoundError:
        registered_test_component = ml_client.components.create_or_update(
            test_component)

    # Create and submit pipeline.
    @pipeline(default_compute=COMPUTE_NAME,
              experiment_name=EXPERIMENT_NAME,
              display_name="train_test_fashion_mnist")
    def pipeline_func(data_dir: Input) -> Dict:
        train_job = registered_train_component(data_dir=data_dir)
        test_job = registered_test_component(  # pylint: disable=unused-variable
            data_dir=data_dir,
            model_dir=train_job.outputs.model_dir)

        return {
            "model_dir": train_job.outputs.model_dir,
        }

    pipeline_job = pipeline_func(
        data_dir=Input(type="uri_folder", path=registered_dataset.id))

    pipeline_job = ml_client.jobs.create_or_update(pipeline_job)
    ml_client.jobs.stream(pipeline_job.name)

    # Create the model.
    try:
        ml_client.models.get(name=MODEL_NAME, version=MODEL_VERSION)
    except ResourceNotFoundError:
        model_path = f"azureml://jobs/{pipeline_job.name}/outputs/model_dir"
        model = Model(path=model_path,
                      name=MODEL_NAME,
                      version=MODEL_VERSION,
                      type=AssetTypes.MLFLOW_MODEL)
        ml_client.models.create_or_update(model)

    # Download the model (this is optional)
    # Blocked by bug.
    # ml_client.models.download(name=MODEL_NAME, version=MODEL_VERSION)


if __name__ == "__main__":
    main()
