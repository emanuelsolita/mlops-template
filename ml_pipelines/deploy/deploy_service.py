#!/usr/bin/env python3
from azureml.core.model import InferenceConfig
from azureml.core import Workspace, Model
from ml_pipelines.utils import EnvironmentVariables, get_environment, config_compute

from azureml.core.webservice import LocalWebservice, AksWebservice

workspace = Workspace.from_config()
env_vars = EnvironmentVariables()

environment = get_environment(workspace, env_vars)
inference_config = InferenceConfig(
    entry_script=env_vars.scoring_file,
    source_directory=env_vars.scoring_dir,
    environment=environment,
)


# Will return the latest model version
model = Model(workspace, name=env_vars.model_name, version=None)

# Create 
#deployment_config = LocalWebservice.deploy_configuration(port=6789)                        # Local Webservice
deployment_config = AksWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)        # Deployed in AML

inference_cluster_name = "my-ak-2"
aks_target = config_compute(workspace, inference_cluster_name) 


service = Model.deploy(
    workspace=workspace,
    name="cool-deployed-service-2",
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
    overwrite=True,
    deployment_target=aks_target
)
service.wait_for_deployment(show_output=True)
print('uri', service.scoring_uri)
print('key', service.get_keys()[0]) # Your service is per default protected by key authentication



