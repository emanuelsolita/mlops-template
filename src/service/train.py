import pandas as pd

from azureml.core import Workspace, Experiment, Dataset, Datastore, Run, Model
from azureml.exceptions import UserErrorException, WebserviceException
from azureml.core import Run
from pathlib import Path
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import joblib

#ws = Workspace.from_config()
#experiment = Experiment(ws, "diamond-regression-experiment-interactive")
run = Run.get_context()
experiment = run.experiment
ws = run.experiment.workspace

data_path = "../data"
ds_train_name = "diamonds-train.csv"
ds_test_name = "diamonds-test.csv"


try:
    diamond_train_dataset = Dataset.get_by_name(ws, name=ds_train_name, version=None)
    diamonds_train = diamond_train_dataset.to_pandas_dataframe()
except UserErrorException:
    print(f"Failed to find a dataset by this name {ds_train_name}. Try to register a new one")
    diamonds_train = pd.read_csv(Path(data_path, 'diamonds-train.csv'))
    datastore = Datastore.get_default(ws)
    # Register the dataset
    diamond_train_dataset = Dataset.Tabular.register_pandas_dataframe(
        diamonds_train,
        datastore, 
        show_progress=True, 
        name=ds_train_name, 
        description='Diamond Training Dataset'
    )


try:
    diamond_test_dataset = Dataset.get_by_name(ws, name=ds_test_name, version=None)
    diamonds_test = diamond_test_dataset.to_pandas_dataframe()
except UserErrorException:
    print(f"Failed to find a dataset by this name: {ds_test_name}. Try to register a new one")
    diamonds_test = pd.read_csv(Path(data_path, 'diamonds-test.csv'))
    datastore = Datastore.get_default(ws)
    # Register the dataset
    diamond_test_dataset = Dataset.Tabular.register_pandas_dataframe(
        diamonds_test,
        datastore, 
        show_progress=True, 
        name=ds_test_name, 
        description='Diamond Training Dataset'
    )


def clean_dataframe(df):
    df = df.copy()

    # Filter out the zero values we observed before
    df = df[~((df['x'] == 0) | (df['y'] == 0) | (df['z'] == 0))]
    df = df.dropna()
    return df


from sklearn.model_selection import train_test_split
df_train = clean_dataframe(diamonds_train)
df_test = clean_dataframe(diamonds_test)
print("Size of the training set", len(df_train))
print("Size of the test set", len(df_test))


def prepare_data(df):
    df = df.copy()
    y = df.pop('price')
    return df, y

regressor = LinearRegression()
ct = make_column_transformer(
    (MaxAbsScaler(), make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(), make_column_selector(dtype_include=object)),
)

X_train, y_train = prepare_data(df_train)
X_test, y_test = prepare_data(df_test)

model = Pipeline([("ColumnTransformer", ct), ("Regressor", regressor)])
model.fit(X_train, y_train)
y_ = model.predict(X_test)

rmse = mean_squared_error(y_test, y_, squared=False)
r2 = r2_score(y_test, y_)
print("rmse", rmse)
print("r2", r2)


from pathlib import Path
import joblib

path = Path("outputs", "model.pkl")
path.parent.mkdir(exist_ok=True)
joblib.dump(model, filename=str(path))

#exp = Experiment(ws, 'diamond-regression-experiment-interactive')

#run = exp.start_logging(snapshot_directory=".")  # display_name="My Run"

#run.log('r2', r2)
#run.log('rmse', rmse)


#run.upload_file(str(path.name), path_or_stream=str(path))

#run.complete()