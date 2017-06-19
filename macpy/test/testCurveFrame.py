import macpy.utils.database as db
   

def testSQLStatement():
    server = 'Elara'
    dateRange = '2011-01-01:2012-01-01'
    curveName = 'USD.ISSR(IBM).SPR'
    commands = 'includebonds'
    
    sqlParams = db.CurveParams(server=server, dateRange=dateRange, curveName=curveName, commands=commands)
    sqlStatement = sqlParams.create_statement()
