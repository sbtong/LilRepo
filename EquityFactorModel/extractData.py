#! /usr/bin/python

import sys
from optparse import OptionParser
import os
import os.path
import time

import numpy as np
import pandas

import datamodel
import datamodel.analytics as analytics

def runmain(argv=None):
    if argv == None:
        argv = sys.argv

    usage = 'usage: %prog [options]\n'
    parser = OptionParser(usage=usage)
    parser.add_option("--rm", dest="rmname", default='US3AxiomaMH',
          help='Name of desired riskmodel family (default: %default)')    
    parser.add_option("--dbhost", dest="dbhost", default='localhost',
         help='Host running 7.x database (default: %default)')
    parser.add_option("--dbname", dest="dbname", default='axiomadb_us',
         help='Name of 7.x database (default: %default)')
    parser.add_option("--dbtype", dest="dbType", default='postgresql',
            help='Name of 7.x database type: postgresql or sqlserver (default: %default)')
    parser.add_option("--dir", dest="outdir", default='rmm_data',
            help='Name of output directory (default: %default)')
    (cmdoptions, args) = parser.parse_args(argv)
    
    dataprov = datamodel.DataController(cmdoptions.dbname, cmdoptions.dbhost, dbType=cmdoptions.dbType)

    dates = sorted(dataprov.getRiskModelDates(cmdoptions.rmname))

    if not os.path.exists(cmdoptions.outdir):
        os.makedirs(cmdoptions.outdir)

    for d in dates:
        t0 = time.time()
        rm = dataprov.getRiskModel(d, cmdoptions.rmname)
        estu = rm.estuniv
        rm = analytics.RiskModel.fromDBModel(rm)
        B = rm.B.copy()
        B[B == 0.0] = np.nan
        B['estu'] = pandas.Series(1.0, index=list(estu))
        assets = set(B.index)
        r = dataprov.getAttribute(d, 'market.1-day return', assets=assets)
        mcap = dataprov.getConvertedAttribute(d, 'market.Market Cap', 'USD', assets=assets)
        B['rets'] = pandas.Series(r)
        B['mcap'] = pandas.Series(mcap)
        B.to_csv(os.path.join(cmdoptions.outdir, 'data' + str(d) + '.csv'))
        t1 = time.time()
        print d, t1 - t0


if __name__ == "__main__":
    runmain()
