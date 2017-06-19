import cx_Oracle
import argparse
import datetime
import subprocess
import os
import logging.config
import logging
import sys

import pandas as pd

from macpy.filters import BondFilter, BondFilterDbWriter

from macpy.compare_filter import Compare

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sid", dest="sid", help="Example: -s GLPROD", default="GLPROD")
    parser.add_argument("-u", "--user", dest="user", help="Example: -u modeldb_global", default="modeldb_global")
    parser.add_argument("-p", "--password", dest="password", help="Example: -p modeldb_global", default="modeldb_global")
    parser.add_argument("-c", "--config", dest="configFile", help="Example: -c production.config", default='database.config')
    parser.add_argument("-e", "--env", dest="env", help="Example: -e environment [DEV|PROD]", default='DEV')
    parser.add_argument("-l", "--log-config", dest="logConfig", help="Example: -l log.config", default='/home/ops-rm/global/scripts/Phoenix/log.config')
    parser.add_argument("-t", "--production_table", action="store_true", dest="productionTable", help="store results into production instead of research")
    parser.add_argument("-C", "--config-file", dest="dbConfigFile", help="Example: -C production.config", default='production.config')
    parser.add_argument("-d", "--dir", dest="wd", help="Example: -d /home/ops-rm/global/scripts/Phoenix", default='/home/ops-rm/global/scripts/Phoenix')
    parser.add_argument("-b", "--bootstrap", dest="bootstrap", help="Example: -b", default=False)
    parser.add_argument("-x", "--debug", dest="debug", help="Printout debug info", default=False)
    parser.add_argument("country",  help="country name", metavar="COUNTRY")
    parser.add_argument("startDate",  help="start date", metavar="Start Date")
    parser.add_argument("endDate",  help="start date", metavar="End Date")
    args = parser.parse_args()

    logging.config.fileConfig(args.logConfig)
    logger = logging.getLogger('root')

    # tradeDate= datetime.date(int(args.tradeDate[0:4]), int(args.tradeDate[5:7]), int(args.tradeDate[8:10]))
    start = datetime.datetime.strptime(args.startDate, "%Y-%m-%d")
    end = datetime.datetime.strptime(args.endDate, "%Y-%m-%d")
    date_range = pd.bdate_range(start, end)
    conn = cx_Oracle.connect(args.user, args.password, args.sid)

    query = """
    SELECT currency_code
    FROM RMG_CURRENCY rc
    JOIN risk_model_group rmg ON rmg.rmg_id=rc.rmg_id
    WHERE rmg.mnemonic=:ctry -- AND rc.from_dt<=:dt AND rc.thru_dt>:dt
    --MarketData curves table does not support historical currency codes
    ORDER BY thru_dt DESC"""

    cursor=conn.cursor()
    cursor.execute(query, ctry=args.country)
    currency = cursor.fetchall()[0][0]

    for tradeDate in date_range:
        logger.info('tradeDate:{}, currency:{}, country:{}'.format(tradeDate, currency, args.country))

        env = os.environ
        env['PYTHONPATH']='..:/home/ops-rm/global/scripts/Phoenix'
        bf = BondFilter(currency, args.country, tradeDate, database=args.env, logger=logger,  debug=args.debug)
        df = bf.run_filter()
        if df is None:
            print 'No {}.{} Bonds for TradeDate {}\n'.format(args.country, currency, tradeDate)
        #print 'Filtered Bonds for {}.{}, on {}\n'.format(args.country, currency, tradeDate)
        #print df
        dbconn = BondFilterDbWriter(tradeDate, bf.curveId, df, enviroment=args.env, config=args.configFile, production_table=args.productionTable)
        error_code = dbconn.write_to_dbs()
        if error_code == 1:
            print 'Error bond filter failed'
            sys.exit(1)
        else:
            pass

        if args.country == 'DE':
            EURO_CURVES = [ 
                'AT.EUR.GVT.ZC',
                'BE.EUR.GVT.ZC',
                'CY.EUR.GVT.ZC',
                 #'DE.EUR.GVT.ZC',
                'EP.EUR.GVT.ZC',
                'ES.EUR.GVT.ZC',
                'FI.EUR.GVT.ZC',
                'FR.EUR.GVT.ZC',
                'GR.EUR.GVT.ZC',
                'IE.EUR.GVT.ZC',
                'IT.EUR.GVT.ZC',
                'LT.EUR.GVT.ZC',
                'MT.EUR.GVT.ZC',
                'NL.EUR.GVT.ZC',
                'PT.EUR.GVT.ZC',
                'SK.EUR.GVT.ZC']
            for curve_name in EURO_CURVES:
                s = curve_name.split('.')
                country = s[0]
                currency = 'EUR'
                try:
                    logger.info('tradeDate:{}, currency:{}, country:{}'.format(tradeDate, currency, country))

                    bf = BondFilter(currency, country, tradeDate, debug=False, database=args.env, logger=logger)
                    df = bf.run_filter()
                    print 'Filtered Bonds for {}.{}, on {}\n'.format(country, currency, tradeDate)
                    if df is None:
                        print 'No {}.{} Bonds for TradeDate {}\n'.format(args.country, currency, tradeDate)
                        continue
                    print df
                    dbconn = BondFilterDbWriter(tradeDate, bf.curveId, df, enviroment=args.env, config=args.configFile, production_table=args.productionTable)
                    error_code = dbconn.write_to_dbs()
                    if error_code == 1:
                        print 'Error bond filter failed for {}'.format(country)
                    else:
                        pass
                except Exception, e:
                    logger.error(e)
                    logger.info('FAIL: tradeDate:{}, currency:{}, country:{}'.format(tradeDate, currency, country))
        # error_code = Compare(tradeDate, currency, args.env, args.configFile).run()

        # if error_code == 1:
        #     print 'Legacy and New filter do not match'
        #     sys.exit(1)
        # else:
        #     sys.exit(0)

if __name__ == '__main__':
    main()

