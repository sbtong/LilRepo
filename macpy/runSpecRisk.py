from spec_risk import SpecRisk, SpecRiskDB, get_history

import math
import numpy as np
import pandas as pd
import datetime
import functools
import getpass
import argparse

import macpy
import macpy.utils.database as db
import macpy.bond as bond

from scipy.interpolate import interp1d

import string

import logging.config
import logging

import sys


run_list = [['BGN', 'BG'],['BRL', 'BR'],['CLP', 'CL'],['CNY', 'CN'],['COP', 'CO'],['CZK', 'CZ'],['DKK', 'DK'],['HKD', 'HK'],['HRK', 'HR'],['HUF', 'HU'],['ILS', 'IL'],['INR', 'IN'],['ISK', 'IS'],['KRW', 'KR'],['MXN', 'MX'],['MYR', 'MY'],['NZD', 'NZ'],['PKR', 'PK'],['PLN', 'PL'],['RON', 'RO'],['RUB', 'RU'],['SGD', 'SG'],['THB', 'TH'],['TRY', 'TR'],['TWD', 'TW'],['UAH', 'UA'],['ZAR', 'ZA'],['USD', 'AR'],['USD', 'BR'],['USD', 'CL'],['USD', 'CO'],['USD', 'HR'],['USD', 'HU'],['USD', 'ID'],['USD', 'JM'],['USD', 'LB'],['USD', 'MX'],['USD', 'PA'],['USD', 'PE'],['USD', 'PH'],['USD', 'PK'],['USD', 'PL'],['USD', 'RU'],['USD', 'TR'],['USD', 'UY'],['USD', 'VE'],
['USD', 'ZA'], ['EUR', 'GR']]

def run_spec_risk(params):
    if params['currency'] == 'USD':
        curve_type = 'RsdSprVol.EmgHrdCcy'
    else:
        curve_type = 'RsdSprVol.EmgLclCcy'
    dbconn = db.AxiomaSpreadID(params['currency'], params['country'], params['database'], params['config'])
    curve_node_id = dbconn.extract_from_db().values[0][0]
    dbconn = db.AxiomaCurveID(curve_type=curve_type, database=params['database'], config=params['config'])
    df_curves = dbconn.extract_from_db()
    curve = df_curves[df_curves['CountryEnum'] == params['country']]
    params['curve_id'] = curve['CurveId'].values[0]
    logging.info('Running Residual Spread Vol calc for Country:{}, Currency:{}, CurveNodeID: {}'.format(params['country'], params['currency'], curve_node_id))
    df_hist = get_history(params['country'], 
                          params['currency'], 
                          params['curve_id'],
                          params['trade_date'])
    params['df_deriv'] = df_hist.copy(deep=True)
    sr = SpecRisk(**params)
    sr.run()
    if sr.median is None:
        raise Exception('Error no data: {}.{}'.format(params['currency'], params['country']))
    if math.isnan(sr.median):
        print df_hist
        raise Exception('Error Nan Values: {}.{}'.format(params['currency'], params['country']))
    sp = SpecRiskDB(params['trade_date'],
                    curve_node_id, 
                    sr.median,
                    production_table=True,
                    enviroment=params['database'],
                    config=params['config'])
    sp.write_to_db()
    logging.info('finished: {}.{}'.format(params['currency'], params['country']))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="configFile", help="Example: -c production.config", default='database.config')
    parser.add_argument("-e", "--env", dest="env", help="Example: -e environment [DEV|PROD]", default='DEV')
    parser.add_argument("-t", "--production_table", action="store_true", dest="productionTable", help="store results in Curve Node Quotes instead of research")
    parser.add_argument("-C", "--config-file", dest="dbConfigFile", help="Example: -C production.config", default='production.config')
    parser.add_argument("trade_date",  help="trade_date", metavar="trade_date")
    parser.add_argument("-l", "--log-config", dest="logConfig", help="Example: -l log.config", default='/home/ops-rm/global/scripts/Phoenix/log.config')
    args = parser.parse_args()

    logging.config.fileConfig(args.logConfig)
    error_code = True

    params = {}
    params['trade_date'] = args.trade_date
    params['database'] = args.env
    params['config'] = args.configFile

    for l in run_list:
        params['currency'] = l[0]
        params['country'] = l[1]
        try:
            run_spec_risk(params)
        except Exception, e:
            logging.info('failed: {}.{}'.format(params['currency'], params['country']))
            logging.info(e)
            # error_code = False
    if error_code:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
