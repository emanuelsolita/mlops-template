from azureml.core import Environment, Workspace, Experiment
from ml_pipelines.utils import EnvironmentVariables, get_environment
from azureml.core import ScriptRunConfig

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

environment.python.user_managed_dependencies = True
    


src = ScriptRunConfig(source_directory='src\service', script='train.py', environment=environment)
run = experiment.submit(src)
run.wait_for_completion(show_output=True)
