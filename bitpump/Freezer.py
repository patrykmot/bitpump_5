import configparser
import shutil
import Utils


class FreezerConfiguration:
    def __init__(self):
        self.simulation_name: str = ""
        self.model_error: str = "9999999999"


def remove_all_cached_data():
    if Utils.is_file_exist(Freezer.BASE_FOLDER):
        shutil.rmtree(Freezer.BASE_FOLDER)


class Freezer:
    BASE_FOLDER: str = "Freezer"
    STOCK_BASE_FOLDER = "Stock_data"

    MODEL_FILE_NAME: str = "Model"
    MODEL_ERROR_FILE_NAME = "ModelError.csv"
    MODEL_TRAINING_IN_DESCRIBE = "ModelTrainingInDesc.txt"
    MODEL_TRAINING_TARGET_DESCRIBE = "ModelTrainingTargetDesc.txt"
    MODEL_VALIDATION_TARGET_DESCRIBE = "ModelValidationTargetDesc.txt"
    MODEL_VALIDATION_IN_DESCRIBE = "ModelValidationInDesc.txt"

    CONFIG_FILE_NAME: str = "FreezerConfiguration.txt"
    CONFIG_SECTION: str = "Freezer"
    CONFIG_SECTION_NAME: str = "SimulationName"

    def __init__(self, freezer_id: str, name: str):
        self._freezer_id = freezer_id
        self._folder = Utils.join_file_system_path(Freezer.BASE_FOLDER, freezer_id)
        self._folder_stock_data = Utils.join_file_system_path(self._folder, Freezer.STOCK_BASE_FOLDER)
        self._config_file = self._get_file_path(Freezer.CONFIG_FILE_NAME)
        self._configuration = FreezerConfiguration()

        print(f"Trying to create folder {self._folder}")
        if Utils.create_folder(self._folder):
            # Save new config to file
            print(f"Folder {self._folder} created")
            # Save configuration
            self._configuration.simulation_name = name
            self._save_configuration()
        else:
            # Load config from file
            self._load_configuration()

        # Try to create stock data folder
        Utils.create_folder(self._folder_stock_data)

    def _get_file_path(self, file_name: str):
        return Utils.join_file_system_path(self._folder, file_name)

    def _save_configuration(self):
        print(f"Save freezer configuration to file: {self._config_file}")
        self._configuration.__dict__
        parser = configparser.ConfigParser()
        parser.add_section(Freezer.CONFIG_SECTION)
        for k, v in self._configuration.__dict__.items():
            parser.set(Freezer.CONFIG_SECTION, k, v)
        with open(self._config_file, "w") as configfile:
            parser.write(configfile)

    def _load_configuration(self):
        parser = configparser.ConfigParser()
        if Utils.is_file_exist(self._config_file):
            print(f"Load freezer configuration from file: {self._config_file}")
            parser.read(self._config_file)
            for k, v in self._configuration.__dict__.items():
                property_value = parser.get(Freezer.CONFIG_SECTION, k)
                self._configuration.__setattr__(k, property_value)

    def get_model_data(self) -> (str, str):
        return self._get_file_path(Freezer.MODEL_FILE_NAME), float(self._configuration.model_error)

    def save_model_data(self, error: float):
        self._configuration.model_error = str(error)
        self._save_configuration()

    def is_stock_file_exist(self, file_name):
        return Utils.is_file_exist(Utils.join_file_system_path(self._folder_stock_data, file_name))

    def get_stock_file_path(self, file_name):
        return Utils.join_file_system_path(self._folder_stock_data, file_name)

    def get_error_file(self):
        return self._get_file_path(Freezer.MODEL_ERROR_FILE_NAME)

    def get_model_training_describe_file_names(self):
        return self._get_file_path(Freezer.MODEL_TRAINING_IN_DESCRIBE), \
            self._get_file_path(Freezer.MODEL_TRAINING_TARGET_DESCRIBE)

    def get_model_validation_describe_file_names(self):
        return self._get_file_path(Freezer.MODEL_VALIDATION_TARGET_DESCRIBE), \
            self._get_file_path(Freezer.MODEL_VALIDATION_IN_DESCRIBE)

