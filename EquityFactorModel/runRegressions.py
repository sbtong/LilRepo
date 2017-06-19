#! /usr/bin/python

import sys
from optparse import OptionParser
import glob
import os.path
import itertools

import numpy as np
import pandas
import statsmodels.api as sm

def computeFactorReturns(df0, df1):
    r = df1['rets'].fillna(0.0)
    estu = df1['estu']
    estu = estu[estu == 1.0].index
    estu = df0.index.intersection(estu)
    B = df0.drop(['rets', 'mcap', 'estu'], axis=1).fillna(0.0)
    result = sm.WLS(r[estu], B.loc[estu], weights=np.sqrt(df0['mcap'].loc[estu])).fit()
    fr = result.params
    assets = r.index.intersection(B.index)
    resid = r[assets] - B.loc[assets].dot(fr)
    return (fr, resid)

def runmain(argv=None):
    if argv == None:
        argv = sys.argv

    usage = 'usage: %prog [options]\n'
    parser = OptionParser(usage=usage)
    parser.add_option("--in", dest="inputdir", default='rmm_data',
            help='Name of input directory (default: %default)')
    parser.add_option("--out", dest="outputdir", default='rmm_result',
            help='Name of input directory (default: %default)')
    (cmdoptions, args) = parser.parse_args(argv)

    files = glob.glob(os.path.join(cmdoptions.inputdir, 'data*.csv'))
    
    if not os.path.exists(cmdoptions.outputdir):
        os.makedirs(cmdoptions.outputdir)
    
    dates = sorted(pandas.to_datetime(x[-14:-4]).date() for x in files)
    for d0, d1 in itertools.izip(dates[:-1],dates[1:]):
        print d0, d1
        df0 = pandas.DataFrame.from_csv(os.path.join(cmdoptions.inputdir, 'data' + str(d0) + '.csv'))
        df1 = pandas.DataFrame.from_csv(os.path.join(cmdoptions.inputdir, 'data' + str(d1) + '.csv'))
        (fr, sr) = computeFactorReturns(df0, df1)
        fr.to_csv(os.path.join(cmdoptions.outputdir, 'fr' + str(d1) + '.csv'))
        sr.to_csv(os.path.join(cmdoptions.outputdir, 'sr' + str(d1) + '.csv'))



if __name__ == "__main__":
    runmain()

