"""Creates and runs an Azure ML pipeline."""

import logging
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import (AmlCompute, CommandComponent, Data,
                                  Environment, Model)
from azure.identity import DefaultAzureCredential

from common_cloud import MODEL_NAME

COMPUTE_NAME = "cluster-cpu"
DATA_NAME = "data-fashion-mnist"
DATA_PATH = Path(Path(__file__).parent.parent, "data")
COMPONENT_TRAIN_NAME = "component_pipeline_sdk_train"
COMPONENT_TEST_NAME = "component_pipeline_sdk_test"
COMPONENT_CODE = Path(Path(__file__).parent.parent, "src")
ENVIRONMENT_NAME = "environment-pipeline-sdk"
CONDA_PATH = Path(Path(__file__).parent, "conda.yml")
EXPERIMENT_NAME = "aml_pipeline_sdk"
DOWNLOADED_MODEL_PATH = Path(Path(__file__).parent.parent)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)

    # Create the compute cluster.
    cluster_cpu = AmlCompute(
        name=COMPUTE_NAME,
        type="amlcompute",
        size="Standard_DS4_v2",
        location="westus",
        min_instances=0,
        max_instances=4,
    )
    ml_client.begin_create_or_update(cluster_cpu)

    # Create the data set.
    dataset = Data(
        name=DATA_NAME,
        description="Fashion MNIST data set",
        path=DATA_PATH.as_posix(),
        type=AssetTypes.URI_FOLDER,
    )
    registered_dataset = ml_client.data.create_or_update(dataset)

    # Create environment for components. We won't register it.
    environment = Environment(name=ENVIRONMENT_NAME,
                              image="mcr.microsoft.com/azureml/" +
                              "openmpi4.1.0-ubuntu20.04:latest",
                              conda_file=CONDA_PATH)

    # Create the components.
    train_component = CommandComponent(
        name=COMPONENT_TRAIN_NAME,
        inputs=dict(data_dir=Input(type="uri_folder"),),
        outputs=dict(model_dir=Output(type="mlflow_model")),
        environment=environment,
        code=COMPONENT_CODE.as_posix(),
        command="python train.py --data_dir ${{inputs.data_dir}} " +
        "--model_dir ${{outputs.model_dir}}",
    )

    test_component = CommandComponent(
        name=COMPONENT_TEST_NAME,
        inputs=dict(data_dir=Input(type="uri_folder"),
                    model_dir=Input(type="mlflow_model")),
        environment=environment,
        code=COMPONENT_CODE.as_posix(),
        command="python test.py --model_dir ${{inputs.model_dir}}")

    registered_train_component = ml_client.components.create_or_update(
        train_component)

    registered_test_component = ml_client.components.create_or_update(
        test_component)

    # Create and submit pipeline.
    @pipeline(experiment_name=EXPERIMENT_NAME, default_compute=COMPUTE_NAME)
    def pipeline_func(data_dir: Input) -> dict[str, str]:
        train_job = registered_train_component(data_dir=data_dir)
        # Ignoring pylint because "test_job" shows up in the Studio UI.
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
    model_path = f"azureml://jobs/{pipeline_job.name}/outputs/model_dir"
    model = Model(name=MODEL_NAME,
                  path=model_path,
                  type=AssetTypes.MLFLOW_MODEL)
    registered_model = ml_client.models.create_or_update(model)

    # Download the model (this is optional).
    ml_client.models.download(name=MODEL_NAME,
                              download_path=DOWNLOADED_MODEL_PATH,
                              version=registered_model.version)


if __name__ == "__main__":
    main()
