from azureml.core import Workspace, Model
from azureml.core.model import InferenceConfig
from ml_pipelines.utils import EnvironmentVariables, get_environment
# ...
'''
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

print(model)
'''
import ml_pipelines.deploy.package_service

'''
# Will return the latest model version
model = Model(workspace, name=env_vars.model_name, version=None)
package = Model.package(
    workspace,
    models=[model],
    inference_config=inference_config,
    generate_dockerfile=True
)

package.wait_for_creation(show_output=True)
package.save("./imagefiles")
'''