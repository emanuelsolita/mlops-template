#!/usr/bin/env python3
import os
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional

from azureml.core import Environment, Workspace
from azureml.core.compute import ComputeTarget, AksCompute
from azureml.core.compute_target import ComputeTargetException
# ...

@dataclass(frozen=True)
class EnvironmentVariables:
    # Load the system environment variables
    load_dotenv()
    model_name: Optional[str] = os.environ.get("MODEL_NAME", "my-regressor-model")
    environment_name: Optional[str] = os.environ.get(
        "ENVIRONMENT_NAME", "conda-environment"
    )
    environment_file: Optional[str] = os.environ.get(
        "ENVIRONMENT_FILE", "environment_setup/ci_dependencies.yml"
    )
    scoring_dir: Optional[str] = os.environ.get("SCORING_DIR", "src")
    scoring_file: Optional[str] = os.environ.get("SCORING_FILE", "service/score.py")

    # ... add as many environment variables you need


def get_environment(ws: Workspace, env_vars: EnvironmentVariables) -> Environment:
    environment_name = env_vars.environment_name
    assert environment_name is not None
    try:
        env = Environment.get(ws, name=environment_name)
    except Exception:
        assert env_vars.environment_file is not None
        env = Environment.from_conda_specification(
            name=environment_name, file_path=env_vars.environment_file
        )
    return env



def config_compute(ws: Workspace):
    inference_cluster_name = "my-aks"
    try:
        aks_target = AksCompute(ws, name=inference_cluster_name)
    except ComputeTargetException:
        provisioning_config = AksCompute.provisioning_configuration(
            vm_size='Standard_D2as_v4', # The smallest size
            agent_count = 1,
            cluster_purpose = AksCompute.ClusterPurpose.DEV_TEST # Needed for having less than three nodes
        )
        aks_target = ComputeTarget.create(
            workspace = ws,
            name = inference_cluster_name,
            provisioning_configuration = provisioning_config
        )
        aks_target.wait_for_completion(show_output = True)

    return aks_target