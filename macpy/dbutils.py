import RiskModels.MarketDB.Utilities as Utilities
import pandas.io.sql as sql
import pandas as pd


class DatabaseConfigurator:
    """
    Adds databaseConfig and environment to an option parser
    """
    @classmethod
    def add_option_argument(cls, optionParser):
        optionParser.add_argument('-e', '--environment', help="Environment to use. Available options: PROD, UAT, DEV. Default is DEV", action="store", default='DEV')
        optionParser.add_argument("databaseConfig", help="Input configuration file containing database connection info.", action="store")
    
    """
    Configure Logger
    """
    @classmethod
    def create_database_connection(cls, filename, environment):
        configFile = open(filename, 'r')
        configuration = Utilities.loadConfigFile(configFile)
        configFile.close()
        cursor = DatabaseConnection(configuration, environment)
        return cursor 

    @classmethod
    def create_database_connection_from_commandline(cls, cmdlineargs):
        return cls.create_database_connection(cmdlineargs.databaseConfig, cmdlineargs.environment)


class DatabaseConnection(object):
    def __init__(self, configuration, environment):
        self.configuration = configuration
        self.environment = environment.replace('\'', '').replace('"', '').strip()
        self.sectionID = 'CurvesDB%s'%(self.environment.upper())
        self.envInfoMap = Utilities.getConfigSectionAsMap(self.configuration, self.sectionID)        
        self.macdb = self.envInfoMap.get('macdb', None)
        self.infoMap = Utilities.getConfigSectionAsMap(self.configuration, self.macdb)
        self.host = self.infoMap.get('host', None)
        self.user = self.infoMap.get('user', None) 
        self.pwd = self.infoMap.get('password', None)
        self.database = self.infoMap.get('database', None)
        self.dbInfo = Utilities.DatabaseInfo(self.host, self.user, self.pwd, self.database)         
    
    def create_dataframe_mssql(self, sqlstatement):
        connection = Utilities.createMSSQLConnection(self.dbInfo) 
        df = sql.read_sql(sqlstatement, connection)
        connection.close()
        return df 

    def create_dictionary_mssql(self, sqlstatement):
        connection = Utilities.createMSSQLConnection(self.dbInfo) 
        cursor = connection.cursor()
        cursor.execute(sqlstatement)
        resultsDB = cursor.fetchall()
        connection.close()
        return resultsDB

def main():
    pass

    
if __name__ == '__main__':
    main()