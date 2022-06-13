"""Creates and runs an Azure ML pipeline."""

import logging
from pathlib import Path
from typing import Dict

from azure.ai.ml import MLClient, Input, load_component
from azure.identity import DefaultAzureCredential
from azure.ai.ml.dsl import pipeline

COMPUTE_NAME = "cluster-gpu"
DATA_NAME = "data-fashion-mnist"
DATA_VERSION = "1"
EXPERIMENT_NAME = "aml-pipeline-sdk"
TRAIN_PATH = Path(Path(__file__).parent, "train.yml")
TEST_PATH = Path(Path(__file__).parent, "test.yml")


def main():
    logging.basicConfig(level=logging.INFO)
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)

    # Making sure the compute exists on Azure ML. If it doesn't, we get an error
    # here.
    ml_client.compute.get(name=COMPUTE_NAME)

    # Getting the data set, which should already be created on Azure ML.
    data = ml_client.data.get(name=DATA_NAME, version=DATA_VERSION)

    # We'll use the components directly, without registering them first.
    train_component = load_component(path=TRAIN_PATH)
    test_component = load_component(path=TEST_PATH)

    # Create and submit pipeline.
    @pipeline(default_compute=COMPUTE_NAME,
              experiment_name=EXPERIMENT_NAME,
              display_name="train_test_fashion_mnist")
    def pipeline_func(data_dir: Input) -> Dict:
        train_job = train_component(data_dir=data_dir)
        test_job = test_component(  # pylint: disable=unused-variable
            data_dir=data_dir,
            model_dir=train_job.outputs.model_dir)

        return {
            "model_dir": train_job.outputs.model_dir,
        }

    pipeline_job = pipeline_func(
        data_dir=Input(type="uri_folder", path=data.id))

    pipeline_job = ml_client.jobs.create_or_update(pipeline_job)
    ml_client.jobs.stream(pipeline_job.name)


if __name__ == "__main__":
    main()
