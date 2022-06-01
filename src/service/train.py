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
import argparse


parser = argparse.ArgumentParser("This")
parser.add_argument('--train-dataset', default='diamonds-train.csv')
parser.add_argument('--test-dataset', default='diamonds-test.csv')
parser.add_argument('--model-name', default='my-regressor-model')
args = parser.parse_args()

train_ds = args.train_dataset
test_ds = args.test_dataset
model_name = args.model_name

#ws = Workspace.from_config()
#experiment = Experiment(ws, "diamond-regression-experiment-interactive")
run = Run.get_context()
experiment = run.experiment
ws = run.experiment.workspace



train_dataset = Dataset.get_by_name(ws, name=train_ds, version=None)
diamonds_train = train_dataset.to_pandas_dataframe()

test_dataset = Dataset.get_by_name(ws, name=test_ds, version=None)
diamonds_test = test_dataset.to_pandas_dataframe()


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

run.parent.log('r2', r2)
run.parent.log('rmse', rmse)

run.upload_file(str(path.name), path_or_stream=str(path))
print("path name")
print(str(path.name))
print("---------")
all_models = Model.list(ws, name=model_name)
if all(r2 > float(model.tags.get("r2", -np.inf)) for model in all_models):
    print("Found a new winner. Registering the model.")
    run.register_model(
        model_name=model_name,
        model_path=str(path.name),
        description="Linear Diamond Regression Model",
        model_framework="ScikitLearn",
        datasets=[("training dataset", train_dataset), ("test dataset", test_dataset)],
        tags={"rmse": rmse, "r2": r2},
    )

