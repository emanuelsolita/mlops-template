from azureml.core import Environment, Workspace, Experiment
from azureml.core import Experiment
from azureml.pipeline.core import PublishedPipeline

pipeline_id = 'f3e71376-99dd-4869-9fd2-a3143cac285f'
pipeline_name = "my-smooth-pipeline"
environment_name = "some-experiment-name"

workspace = Workspace.from_config()
pipeline = PublishedPipeline.get(workspace, id=pipeline_id)

pipelines = PublishedPipeline.list(workspace)
piplines = [p for p in pipelines if p.name == pipeline_name]
pipeline = pipelines[0]

experiment = Experiment(workspace, environment_name)

run = experiment.submit(pipeline)
status = run.wait_for_completion(show_output=True)
print(status) # Should say finished