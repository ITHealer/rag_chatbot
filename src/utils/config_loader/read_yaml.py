import yaml

from pathlib import Path

from src.utils.config import settings
from src.utils.config_loader.config_interface import ConfigReaderInterface


class YamlConfigReader(ConfigReaderInterface):

    def __init__(self):
        super(YamlConfigReader, self).__init__()

    def read_config_from_file(self, config_filename: str):
        conf_path = Path(__file__).joinpath(settings.APP_CONFIG.SETTINGS_DIR, config_filename)
        with open(conf_path) as file:
            config = yaml.safe_load(file)
        return config
