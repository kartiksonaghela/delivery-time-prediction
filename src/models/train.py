import pandas as pd
import yaml
import joblib
import logging
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

TARGET = "time_taken"

# create logger
logger = logging.getLogger("model_training")
logger.setLevel(logging.INFO)

# console handler
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

# add handler to logger
logger.addHandler(handler)

# create a formatter
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


def load_data(data_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error("The file to load does not exist")
    return df


def read_params(file_path):
    with open(file_path, "r") as f:
        params_file = yaml.safe_load(f)
    return params_file


def save_model(model, save_dir: Path, model_name: str):
    save_location = save_dir / model_name
    joblib.dump(value=model, filename=save_location)


def save_transformer(transformer, save_dir: Path, transformer_name: str):
    save_location = save_dir / transformer_name
    joblib.dump(transformer, save_location)


def train_model(model, X_train: pd.DataFrame, y_train):
    model.fit(X_train, y_train)
    return model


def make_X_and_y(data: pd.DataFrame, target_column: str):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


if __name__ == "__main__":
    # root path
    root_path = Path(__file__).parent.parent.parent
    data_path = root_path / "data" / "processed" / "train_trans.csv"
    params_file_path = root_path / "params.yaml"

    # load the training data
    training_data = load_data(data_path)
    logger.info("Training Data read successfully")

    # split the data into X and y
    X_train, y_train = make_X_and_y(training_data, TARGET)
    logger.info("Dataset splitting completed")

    # read random forest parameters
    model_params = read_params(params_file_path)['Train']
    rf_params = model_params['Random_Forest']
    logger.info("Random Forest parameters read")

    # create model and transformer
    rf = RandomForestRegressor(**rf_params)
    power_transform = PowerTransformer()
    logger.info("Random Forest and PowerTransformer initialized")

    # wrap with target transformer
    model = TransformedTargetRegressor(regressor=rf, transformer=power_transform)
    logger.info("Model wrapped with TransformedTargetRegressor")

    # train the model
    train_model(model, X_train, y_train)
    logger.info("Model training completed")

    # create save directory
    model_save_dir = root_path / "models"
    model_save_dir.mkdir(exist_ok=True)

    # save full model
    save_model(model=model, save_dir=model_save_dir, model_name="model.joblib")
    logger.info("Trained model saved to location")

    # save transformer separately
    save_transformer(transformer=power_transform, save_dir=model_save_dir, transformer_name="power_transformer.joblib")
    logger.info("Transformer saved to location")
