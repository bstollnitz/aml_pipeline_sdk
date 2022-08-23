# How to train using an Azure ML pipeline, using the SDK

This project shows how to train a Fashion MNIST model using an Azure ML pipeline, and how to deploy it using an online managed endpoint. It demonstrates how to create Azure ML resources using the Python SDK, and it uses MLflow for tracking and model representation.


## Blog post

To learn more about the code in this repo, check out the accompanying blog post: https://bea.stollnitz.com/blog/aml-pipeline/


## Azure setup

* You need to have an Azure subscription. You can get a [free subscription](https://azure.microsoft.com/en-us/free?WT.mc_id=aiml-42161-bstollnitz) to try it out.
* Create a [resource group](https://docs.microsoft.com/en-us/azure/azure-resource-manager/management/manage-resource-groups-portal?WT.mc_id=aiml-42161-bstollnitz).
* Create a new machine learning workspace by following the "Create the workspace" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources?WT.mc_id=aiml-42161-bstollnitz). Keep in mind that you'll be creating a "machine learning workspace" Azure resource, not a "workspace" Azure resource, which is entirely different!
* If you have access to GitHub Codespaces, click on the "Code" button in this GitHub repo, select the "Codespaces" tab, and then click on "New codespace."
* Alternatively, if you plan to use your local machine:
  * Install the Azure CLI by following the instructions in the [documentation](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?WT.mc_id=aiml-42161-bstollnitz).
  * Install the ML extension to the Azure CLI by following the "Installation" section of the [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-cli?WT.mc_id=aiml-42161-bstollnitz).
* In a terminal window, login to Azure by executing `az login --use-device-code`. 
* Set your default subscription by executing `az account set -s "<YOUR_SUBSCRIPTION_NAME_OR_ID>"`. You can verify your default subscription by executing `az account show`, or by looking at `~/.azure/azureProfile.json`.
* Set your default resource group and workspace by executing `az configure --defaults group="<YOUR_RESOURCE_GROUP>" workspace="<YOUR_WORKSPACE>"`. You can verify your defaults by executing `az configure --list-defaults` or by looking at `~/.azure/config`.
* You can now open the [Azure Machine Learning studio](https://ml.azure.com/?WT.mc_id=aiml-42161-bstollnitz), where you'll be able to see and manage all the machine learning resources we'll be creating.
* Although not essential to run the code in this post, I highly recommend installing the [Azure Machine Learning extension for VS Code](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai).


## Project setup

If you have access to GitHub Codespaces, click on the "Code" button in this GitHub repo, select the "Codespaces" tab, and then click on "New codespace."

Alternatively, you can set up your local machine using the following steps.

Install conda environment:

```
conda env create -f environment.yml
```

Activate conda environment:

```
conda activate aml_pipeline_sdk
```


## Train and predict locally

```
cd aml_pipeline_sdk
```

* Run src/train.py by pressing F5.
* Run src/test.py the same way.
* You can analyze the metrics logged in the "mlruns" directory with the following command:

```
mlflow ui
```

* Make a local prediction using the trained mlflow model. You can use either csv or json files:

```
mlflow models predict --model-uri "model" --input-path "test_data/images.csv" --content-type csv
mlflow models predict --model-uri "model" --input-path "test_data/images.json" --content-type json
```


## Train and deploy in the cloud

Make sure you have a "config.json" file somewhere in the parent folder hierarchy containing your Azure subscription ID, resource group, and workspace:

```
{
    "subscription_id": ...,
    "resource_group": ...,
    "workspace_name": ...
}
```

### Create and run the pipeline

Run cloud/pipeline_job.py.


### Create and invoke the endpoint

Run cloud/endpoint.py.

### Clean up the endpoint

Once you're done working with the endpoint, you can clean it up to avoid getting charged by running cloud/delete_endpoint.py.


## Related resources

* [Component YAML schema](https://docs.microsoft.com/en-us/azure/machine-learning/reference-yaml-component-command?WT.mc_id=aiml-42161-bstollnitz)
* [Pipeline YAML schema](https://docs.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-pipeline?WT.mc_id=aiml-42161-bstollnitz)
* [az ml job commands](https://docs.microsoft.com/en-us/cli/azure/ml/job?view=azure-cli-latest#az-ml-job-create?WT.mc_id=aiml-42161-bstollnitz)
* [Output formats for Azure CLI commands](https://docs.microsoft.com/en-us/cli/azure/format-output-azure-cli?WT.mc_id=aiml-42161-bstollnitz)
