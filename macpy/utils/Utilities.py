
import time
import datetime
import string
import pymssql
#import cx_Oracle
import logging.config
import ConfigParser
from itertools import izip_longest


def grouper(iterable, n, fill_value=None):
    args = [iter(iterable)] * n
    temp_iter = map(lambda x: tuple(v for v in x if v is not None), izip_longest(*args, fillvalue=fill_value))
    return temp_iter

class Struct:
    def __init__(self, copy=None):
        if copy is not None:
            self.__dict__ = dict(copy.__dict__)
    def getFields(self): return self.__dict__.values()
    def getFieldNames(self): return self.__dict__.keys()
    def setField(self, name, val): self.__dict__[name] = val
    def getField(self, name): return self.__dict__[name]
    def __str__(self):
        retval = []
        for (i, j) in self.__dict__.iteritems():
            if isinstance(j, unicode):
                j = j.encode('utf-8')
            retval.append('%s: %s' % (i, j))
        return '(%s)' % ', '.join(retval)
    def __repr__(self):
        return self.__str__()
 
# Dictionary Utility Functions   
def findDictionaryKeys(dic, val):
    """return the key of dictionary dic given the value"""
    return [k for k, v in dic.iteritems() if v == val]

def findDictionaryValue(dic, key):
    """return the value of dictionary dic given the key"""
    return dic[key]


'''
Document Me!
  Abstract base class for Java Type enumerations; more work needed here.
  
  Example: DataType = enum('DOUBLE', 'NUMBER', 'FLOAT', 'INTEGER', 'DATE', 'STRING', 'SHARES')
           DataType2 = enum(DOUBLE='double', NUMBER='number')
           myType = DataType.SHARES 
           myType2 = DataType2.DOUBLE
           
           myType evaluates to ordinal 6
           myType2 evaluates to string 'double'
'''
#def enum(**enums):
#    return type('Enum', (), enums)
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((value, key) for key, value in enums.iteritems())
    enums['reverse_mapping'] = reverse
    return type('Enum', (), enums)

DATA_TYPE = enum(ESTIMATE='estimate', FUNDAMENTAL='fundamental')

COUNTRY = enum(CANADA='Canada', USA='USA', OTHER='Other')

class DatabaseInfo:
    def __init__(self, databaseHost, databaseUser, databasePWD, databaseName, serverPort = None, sid=None):
        #  TODO: Input Validation Missing Here
        self.host = databaseHost
        self.user = databaseUser
        self.password = databasePWD
        self.databaseName = databaseName
        self.port = int(serverPort) if serverPort is not None else 1433 # MSSQL Default Port
        self.sid = sid  # Relevant for Oracle connections only

def loadConfigFile(configFile):
    if configFile is None:
        return configFile
    
    config = ConfigParser.SafeConfigParser()
    config.readfp(configFile)
    return config

def getConfiguration(configurationFilePath):
    if configurationFilePath is None:
        return configurationFilePath
    
    configFile = open(configurationFilePath, 'r')
    configuration = loadConfigFile(configFile)
    configFile.close()
    return configuration

'''
Documentation Missing Here
'''
class DataMatrix():
    '''TODO: Input Validation Missing Here'''
    def __init__(self, columnIDs, dataInfo):
        self.columnIDs = columnIDs
        self.dataInfo = dataInfo
        self.columnCount = len(self.columnIDs)
        self.rowCount = len(dataInfo)
        
    def getColumnCount(self):
        return self.columnCount
    
    def getColumnID(self, columnIndex):
        assert(columnIndex is not None and columnIndex <= self.getColumnCount())
        return self.columnIDs[columnIndex]
    
    def getColumnIDs(self):
        return self.columnIDs
    
    def getColumnIndex(self, columnID):
        assert(columnID is not None and columnID in self.columnIDs)
        return self.columnIDs.index(columnID)
    
    def getDataType(self, columnIndex):
        return self.dataType[columnIndex]
    
    def getValue(self, rowIndex, columnIndex):
        assert(rowIndex is not None and rowIndex < self.getRowCount())
        assert(columnIndex is not None and columnIndex < self.getColumnCount())
        return self.dataInfo[rowIndex][columnIndex]
        
    def getRowCount(self):          
        return self.rowCount
    
    def getRowValues(self, rowIndex):
        assert(rowIndex is not None and rowIndex < self.getRowCount())
        return self.dataInfo[rowIndex]

    def getDataInfo(self):
        return self.dataInfo

'''
   Iterator that parses an iteratable instances and groups its constituents
   based on user criteria (specified via a function):
   Example:
       myList = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
       myValues = {'A' : 'X', 'B' : 'B', 'C' : 'X', 'D': 'B', 'E' : 'Z', 'F' : 'C', 'G' : 'C', 'H' : 'B' }
    
       groupedValues = Utilities.groupBy(myList, lambda x: myValues.get(x), True)
       for key, items in groupedValues:
           values = [s for s in items]
           print 'ITEM[%s]:%s'%(key, values)
'''
class groupBy(object):
    
    def __init__(self, iterableCollection, itemValueFunction, itemSelectionFunction=None, reverseOrder=False):
        self.itemValueFunction = itemValueFunction
        self.itemSelectionFunction = itemSelectionFunction
        
        # Order the collection so that items with same values are contiguous
        self.userOrderedCollectionIter = iter(sorted(iterableCollection, 
                                                     key=lambda x:itemValueFunction(x), 
                                                     reverse=reverseOrder)) 
        # Initialization of member variables
        self.presentUserItem = next(self.userOrderedCollectionIter)
        self.presentItemValue = self.itemValueFunction(self.presentUserItem)
        self.presentTargetValue = None
        
    # Provide implementation for iterator interface so that this class is itself
    # an iterator and can be used in for loop construct
    def __iter__(self):
        return self
        
    def next(self):
        if self.presentTargetValue == self.presentItemValue: # Check if we need to stop (reached end of collection)
            next(self.userOrderedCollectionIter)             # Force Stopping Condition: Exit on StopIteration Exception
         
        #--- Update Present Target Value ---#
        self.presentTargetValue = self.presentItemValue
        return (self.presentTargetValue, self._getUserCollectionSubGroup())

    def _getUserCollectionSubGroup(self):
        while (self.presentTargetValue == self.presentItemValue):
            if self.keepItem(self.presentUserItem):
                yield self.presentUserItem
            self.presentUserItem = next(self.userOrderedCollectionIter)  # Will Stop at Stop Exception
            self.presentItemValue = self.itemValueFunction(self.presentUserItem)
            
    def keepItem(self, item):
        return True if self.itemSelectionFunction is None else self.itemSelectionFunction(item)

def parseISODate(dateStr):
    """Parse a string in YYYY-MM-DD format and return the corresponding
    date object.
    """
    assert(len(dateStr) == 10)
    assert(dateStr[4] == '-' and dateStr[7] == '-')
    return datetime.date(int(dateStr[0:4]), int(dateStr[5:7]),
                         int(dateStr[8:10]))

def addDefaultCommandLine(optionParser):
    """Add options to select the log configuration file.
    """
    optionParser.add_option("-l", "--log-config", action="store",
                            default='log.config', dest="logConfigFile",
                            help="logging configuration file")
    
def processDefaultCommandLine(options, optionsParser):
    """Configure the log system.
    """
    import os
    import sys
    try:
        logging.config.fileConfig(options.logConfigFile)
        if sys.stderr.isatty() and sys.stdout.isatty() and \
           os.ttyname(sys.stdout.fileno()) == os.ttyname(sys.stderr.fileno()):
            for h in logging.root.handlers:
                if hasattr(h, 'stream') and h.stream == sys.stderr:
                    logging.root.removeHandler(h)
                    break
    except Exception, e:
        print e
        optionsParser.error('Error parsing log configuration "%s"'% options.logConfigFile)

def computeSEDOLCheckDigit(sedol):
    """Compute the check-digit for a SEDOL.
    Base on the description at http://en.wikipedia.org/wiki/SEDOL.
    """
    weights = [1, 3, 1, 7, 3, 9, 1]
    sedol6 = sedol[:6]
    checksum = 0
    for (i, w) in zip(sedol6, weights):
        v = ord(i) - ord('0')
        if v > 9:
            v = ord(i) - ord('A') + 10
        checksum += v*w
    return (10-checksum) % 10

def computeCUSIPCheckDigit(cusip):
    """Compute the check-digit for a CUSIP.
    Based on http://en.wikipedia.org/wiki/CUSIP
    """
    #Convert alpha characters to digits
    cusip2 = []
    for char in cusip:
        if char.isalpha():
            cusip2.append((string.ascii_uppercase.index(char.upper()) + 9 + 1))
        else:
            cusip2.append(char)
    
    #Gather every second digit (even)
    even = cusip2[::2]
    
    #Gather the other digits (odd)
    odd = cusip2[1::2]
    
    # Multiply odds by 2 and convert both to a string of numbers
    odd = ''.join([str(int(i)*2) for i in list(odd)])
    even = ''.join([str(int(i)) for i in list(even)])
    
    #then add each single int in both odd and even
    even_sum = sum([int(i) for i in even])
    odd_sum = sum([int(i) for i in odd])
    
    # Return 10's complement of last digit
    return (10-((even_sum + odd_sum)%10))%10

def computeISINCheckDigit(isin):
    """Calculate and return the check-digit for
    an ISIN. Based on http://en.wikipedia.org/wiki/ISIN"""
    
    #Convert alpha characters to digits
    isin2 = []
    for char in isin:
        if char.isalpha():
            isin2.append((string.ascii_uppercase.index(char.upper()) + 9 + 1))
        else:
            isin2.append(char)
    
    #Convert each int into string and join
    isin2 = ''.join([str(i) for i in isin2])
    
    #Gather every second digit (even)
    even = isin2[::2]
    
    #Gather the other digits (odd)
    odd = isin2[1::2]
    try:
        #If len(isin2) is odd, multiply evens by 2, else multiply odds by 2
        if len(isin2) % 2 > 0:
            even = ''.join([str(int(i)*2) for i in list(even)])
        else:
            odd = ''.join([str(int(i)*2) for i in list(odd)])
        even_sum = sum([int(i) for i in even])

        #then add each single int in both odd and even
        odd_sum = sum([int(i) for i in odd])
        mod = (even_sum + odd_sum) % 10
        return (10 - mod)%10
    except Exception, e:
        logging.error("Isin check digit calculation error for %s: %s" % (isin, e))
        return None

def listChunkIterator(myList, chunkSize):
    """Iterator through the given list in chunk of chunkSize.
    The last chunk will be smaller if the lenght of the list is
    not a multiple of the chunk size.
    """
    for idx in xrange(0, len(myList), chunkSize):
        yield myList[idx:(idx+chunkSize)]
        
def adjustTimeStamp(timeStamp, deltaSecs):
    if timeStamp is not None:
        return timeStamp + datetime.timedelta(seconds=deltaSecs)
    else:
        return timeStamp

def createDateList(dateString):
    dates = [d.strip() for d in dateString.split(',')]
    dateList = list()
    for d in dates:
        if d.find(':') != - 1:
            (startDt, endDt) = d.split(':')
            startDt = parseISODate(startDt)
            endDt = parseISODate(endDt)
            if startDt > endDt:
                logging.error('Invalid date range: %s,%s.', startDt, endDt) # Revisit this: Handle gracefully.
                return None
            else:
                oneDay = datetime.timedelta(days=1)
                while startDt <= endDt:
                    dateList.append(startDt)
                    startDt = startDt + oneDay
        else:
            dateList.append(parseISODate(d))
    dateList.sort()
    return dateList

'''Converts date naive (string, date) to aware datetime instance.'''
def getDateTimeInstance(dateIn):
    if isinstance(dateIn, basestring):
        return datetime_from_str(dateIn)[1]
            
    if isinstance(dateIn, datetime.date) and not isinstance(dateIn, datetime.datetime):
        return datetime.datetime.combine(dateIn, datetime.time())

    return dateIn
    
def datetime_from_str(time_str):
    """Return (<scope>, <datetime.datetime() instance>) for the given datetime string.
       datetime_from_str("2009")
          ('year', datetime.datetime(2009, 1, 1, 0, 0))
        datetime_from_str("2009-12")
          ('month', datetime.datetime(2009, 12, 1, 0, 0))
        _datetime_from_str("2009-12-25")
          ('day', datetime.datetime(2009, 12, 25, 0, 0))
         datetime_from_str("2009-12-25 13")
          ('hour', datetime.datetime(2009, 12, 25, 13, 0))
        datetime_from_str("2009-12-25 13:05")
          ('minute', datetime.datetime(2009, 12, 25, 13, 5))
        datetime_from_str("2009-12-25 13:05:14")
          ('second', datetime.datetime(2009, 12, 25, 13, 5, 14))
        datetime_from_str("2009-12-25 13:05:14.453728")
          ('microsecond', datetime.datetime(2009, 12, 25, 13, 5, 14, 453728))
    """
    formats = [
            # <scope>, <pattern>, <format>
            ("year", "YYYY", "%Y"),
            ("month", "YYYY-MM", "%Y-%m"),
            ("day", "YYYY-MM-DD", "%Y-%m-%d"),
            ("hour", "YYYY-MM-DD HH", "%Y-%m-%d %H"),
            ("minute", "YYYY-MM-DD HH:MM", "%Y-%m-%d %H:%M"),
            ("second", "YYYY-MM-DD HH:MM:SS", "%Y-%m-%d %H:%M:%S"),
            # ".<microsecond>" at end is manually handled below
            ("microsecond", "YYYY-MM-DD HH:MM:SS", "%Y-%m-%d %H:%M:%S"),
    ]

    for scope, pattern, format in formats:
        if scope == "microsecond":
            # Special handling for microsecond part. AFAIK there isn't a strftime code for this.
            if time_str.count('.') != 1:
                continue
            time_str, microseconds_str = time_str.split('.')
            try:
                microsecond = int((microseconds_str + '000000')[:6])
            except ValueError:
                continue
        try:
            # This comment here is the modern way. The subsequent two lines are for Python 2.4 support.
            #t = datetime.datetime.strptime(time_str, format)
            t_tuple = time.strptime(time_str, format)
            t = datetime.datetime(*t_tuple[:6])
        except ValueError:
            pass
        else:
            if scope == "microsecond":
                t = t.replace(microsecond=microsecond)
            return scope, t
    else:
        raise ValueError("could not determine date from %r: does not match any of the accepted patterns ('%s')"% 
                        (time_str, "', '".join(s for s, p, f in formats)))

def getConfigSectionAsMap(config, sectionID):
    valueMap = {}
    if config.has_section(sectionID):
        valueDict = config._sections.get(sectionID)
        valueMap = {key : value for (key, value) in valueDict.iteritems() if key !=  '__name__'}
    #else:
        #logging.error("No %s section in config"%sectionID)

    return valueMap


def validateDatabaseInfo(databaseInfo):
    if databaseInfo is None:
        raise Exception('Invalid argument: createDBConnection() databaseInfo cannot be None.')

    if not isinstance(databaseInfo, DatabaseInfo):
        raise Exception('Invalid argument: createDBConnection() databaseInfo is not instance of DatabaseInfo.')

    if databaseInfo.host is None:
        raise Exception('Invalid argument: createDBConnection() databaseInfo.host cannot be None.')
    
    if databaseInfo.user is None:
        raise Exception('Invalid argument: createDBConnection() databaseInfo.user cannot be None.')
    
    if databaseInfo.password is None:
        raise Exception('Invalid argument: createDBConnection() databaseInfo.password cannot be None.')
    
    if databaseInfo.databaseName is None:
        raise Exception('Invalid argument: createDBConnection() databaseInfo.databaseName cannot be None.')


def createMSSQLConnection(databaseInfo):
    validateDatabaseInfo(databaseInfo)
    
    return pymssql.connect(host = databaseInfo.host,
                           user = databaseInfo.user, 
                           password = databaseInfo.password,
                           database = databaseInfo.databaseName)


def createOracleConnection(databaseInfo):
    validateDatabaseInfo(databaseInfo)
    dsn_tns = cx_Oracle.makedsn(databaseInfo.host, databaseInfo.port, databaseInfo.sid)
    return cx_Oracle.connect(databaseInfo.user, databaseInfo.password, dsn_tns)


def createOracleDBConnection(user, password, sid):
    return cx_Oracle.connect(user, password, sid)


def parse_dates(dates):
    date_list = dates.split(':') if ':' in dates else None
    if date_list is not None:
        start_date = None
        end_date = None
        if len(date_list) > 1:
            start_date = date_list[0]
            end_date = date_list[-1]
        else:
            start_date = date_list[0]
            end_date = start_date
        return start_date, end_date
    else:
        return dates, dates

