from azureml.core import Environment, Workspace, Experiment
from ml_pipelines.utils import EnvironmentVariables, get_environment
from azureml.core import ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

ws = Workspace.from_config()
env_vars = EnvironmentVariables()

environment_name = "some-experiment-name"
experiment = Experiment(ws, environment_name)

try:
    environment = Environment.get(ws, environment_name=environment_name)
except Exception as e:
    print("Defining a new environment")
    environment = Environment.from_conda_specification(
        name=environment_name, file_path="environment_setup/ci_dependencies.yml"
    )
    environment.register(ws)

# Uncomment when running locally
#environment.python.user_managed_dependencies = True

cpu_cluster_name = "my-cool-cluster"
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="Standard_D2as_v4",
        max_nodes=1,
        idle_seconds_before_scaledown=1200, # Scale down after 20 minutes
    )
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)
    # This will block the script until the resource is created
    cpu_cluster.wait_for_completion(show_output=True)



src = ScriptRunConfig(
    source_directory='src/service',
    script='train.py', 
    environment=environment,
    compute_target=cpu_cluster
    )
    
run = experiment.submit(src)
run.wait_for_completion(show_output=True)
