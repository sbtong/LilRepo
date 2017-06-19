import cx_Oracle
import argparse
import datetime
import subprocess
import os
import logging.config
import logging
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sid", dest="sid", help="Example: -s GLPROD", default="GLPROD")
    parser.add_argument("-u", "--user", dest="user", help="Example: -u modeldb_global", default="modeldb_global")
    parser.add_argument("-p", "--password", dest="password", help="Example: -p modeldb_global", default="modeldb_global")
    parser.add_argument("-c", "--config", dest="configFile", help="Example: -c production.config", default='database.config')
    parser.add_argument("-e", "--env", dest="env", help="Example: -e environment [DEV|PROD]", default='DEV')
    parser.add_argument("-l", "--log-config", dest="logConfig", help="Example: -l log.config", default='/home/ops-rm/global/scripts/Phoenix/log.config')
    parser.add_argument("-t", "--production_table", action="store_true", dest="productionTable", help="store results in Curve Node Quotes instead of research")
    parser.add_argument("-C", "--config-file", dest="dbConfigFile", help="Example: -C production.config", default='production.config')
    parser.add_argument("-d", "--dir", dest="wd", help="Example: -d /home/ops-rm/global/scripts/Phoenix", default='/home/ops-rm/global/scripts/Phoenix')
    parser.add_argument("country",  help="country name", metavar="COUNTRY")
    parser.add_argument("startDate",  help="start date", metavar="Start Date")
    parser.add_argument("endDate",  help="start date", metavar="End Date")
    args = parser.parse_args()

    logging.config.fileConfig(args.logConfig)

    startDate= datetime.date(int(args.startDate[0:4]), int(args.startDate[5:7]),
                         int(args.startDate[8:10]))
    endDate = datetime.date(int(args.endDate[0:4]), int(args.endDate[5:7]), int(args.endDate[8:10]))
    conn = cx_Oracle.connect(args.user, args.password, args.sid)

    query = """SELECT currency_code FROM RMG_CURRENCY rc JOIN risk_model_group rmg ON rmg.rmg_id=rc.rmg_id
        WHERE rmg.mnemonic=:ctry AND rc.from_dt<=:dt AND rc.thru_dt>:dt"""
    cursor=conn.cursor()
    cursor.execute(query, ctry=args.country, dt=startDate)
    currency =  cursor.fetchall()[0][0]
    logging.info('Currency=%s', currency)

    env = os.environ
    env['PYTHONPATH']='..:/home/ops-rm/global/scripts/Phoenix'
    if args.productionTable:
        prodTable="-t"
    else:
        prodTable=""

    paramDict={'configFile':args.configFile, 'currency': currency, 'start':str(startDate), 'end':str(endDate), 'env':args.env, 'prodTable':prodTable, 'country':args.country}
    cmd="/home/ops-rm/venv/bin/python curve_stats.py -s %(start)s -e %(end)s -g %(configFile)s -d %(env)s -c %(currency)s -q %(country)s %(prodTable)s" % paramDict
    logging.info(cmd)
    subid = subprocess.Popen(cmd, env=env, shell=True, cwd='.')
    (pid, stat) = os.wait()
    if stat:
       errorCase=True
    else:
       errorCase=False

    if errorCase:
       sys.exit(1)
    else:
       sys.exit(0)
 
if __name__ == '__main__':
    main()


