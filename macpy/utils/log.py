"""
  Provides commonly used logging configuration and utilities 
"""

import os
import logging

class LogConfigurator:
    """
    Adds -l|--logConfigFile to an option parser
    """
    @classmethod
    def add_option_argument(cls, optionParser):
        optionParser.add_argument('-l', '--logConfigFile', help="Optional full path to logging configuration file.", action="store")
    
    """
    Configure Logger
    """
    @classmethod
    def configure_logger(cls, commandlineargs):
        logConfigFile = commandlineargs.logConfigFile if commandlineargs.logConfigFile else os.path.dirname(os.path.realpath(__file__))+"/log.config"
        if not os.path.exists(logConfigFile):
            raise Exception("Logging configuration file:%s does not exist."%logConfigFile)
        logging.config.fileConfig(logConfigFile)


def activate_logging(log_config_file_path=None, working_directory=None):
    log_config_file = log_config_file_path if log_config_file_path is not None \
        else os.path.join(working_directory, "log.config")
    if not os.path.exists(log_config_file):
        raise Exception("Logging configuration file:%s does not exist." % log_config_file)

    logging.config.fileConfig(log_config_file)


def create_print_logger():
    def log(x):
        print(x)
    return log


def create_info_logger():
    def log(x):
        logging.info(x)
    return log

def create_null_logger():
    def log(x):
        pass
    return log


def main():
    pass


if __name__ == '__main__':
    main()
