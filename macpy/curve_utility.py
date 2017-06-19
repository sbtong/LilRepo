from utils import Utilities
import datetime
import pandas as pd
#----------------------------#
#     Utility Functions      #
#----------------------------#
def getAnalysisConfigSection(environmentName):
    if environmentName is not None:
        env = environmentName.replace('\'', '').replace('"', '').strip()
        return 'CurvesDB%s'%(env.upper())
    else:
        return environmentName

def getOracleDatabaseInfo(infoMap):
    sid = infoMap.get('sid', None)
    port = infoMap.get('port', 1600)  # Default Value 
    dbID = sid  # Use this for now -- Has no impact
    return Utilities.DatabaseInfo(infoMap.get('host', None), infoMap.get('user', None), 
                                  infoMap.get('password', None), dbID, port, sid)

def getMSSQLDatabaseInfo(infoMap):
    return Utilities.DatabaseInfo(infoMap.get('host', None), infoMap.get('user', None), 
                                  infoMap.get('password', None), infoMap.get('database', None))

def add_business_days(date, days=0):
    _d = pd.to_datetime(date)
    if date.weekday() == 4: #Friday
        _d =(_d + datetime.timedelta(days=3)).date()
        return (pd.to_datetime(_d) + datetime.timedelta(days=days-1)).date()
    elif date.weekday() == 5: #Saturday
        _d =(pd.to_datetime(_d) + datetime.timedelta(days=2)).date()
        return (pd.to_datetime(_d) + datetime.timedelta(days=days-1)).date()
    return (_d + datetime.timedelta(days=days)).date()
    