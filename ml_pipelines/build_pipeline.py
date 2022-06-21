from azureml.core import Environment, Workspace, Experiment
from ml_pipelines.utils import EnvironmentVariables, get_environment
from azureml.core import ScriptRunConfig, RunConfiguration
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineParameter

ws = Workspace.from_config()
env_vars = EnvironmentVariables()

environment_name = "some-environment-name"
experiment_name = "remote-experiment-name"
experiment = Experiment(ws, experiment_name)

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

run_config = RunConfiguration()
# Remember to set our favorite environment
run_config.environment = environment

parameters = [PipelineParameter('train-dataset', 'diamonds-train.csv'),
              PipelineParameter('test-dataset', 'diamonds-test.csv'),
              PipelineParameter('model-name', 'my-regressor-model')
             ]

train_step = PythonScriptStep(
    name="training_step",
    script_name="train.py",
    source_directory="src/service",
    compute_target=cpu_cluster,
    runconfig=run_config,
    allow_reuse=False,
    arguments = ['--train-dataset',parameters[0],'--test-dataset',parameters[1],'--model-name',parameters[1]]
)

pipeline = Pipeline(
    workspace=ws, steps=[train_step], description="Model Training and Deployment"
)
pipeline.validate() # Make sure the pipeline is functioning

pipeline_name = "my-smooth-pipeline"
published_pipeline = pipeline.publish(pipeline_name)
print(published_pipeline.id)


