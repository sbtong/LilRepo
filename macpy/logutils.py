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


def main():
    pass

if __name__ == '__main__':
    main()