
import os
import pandas as pd
dir_path = os.path.dirname(os.path.realpath(__file__))
import next_gen_curve as ngc
from string import Template
import macpy.utils.database as db
import logging
import utils.ngc_utils as u

def main():
    logging.basicConfig(filename='ngc.log', level=logging.INFO, filemode='w',format='%(asctime)s %(levelname)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    currencyList = ['CAD']
    #if only one curve
    fitSingle = True
    specific_curve = 'USD-VENZ-SOV'
    if fitSingle:
        Currency = specific_curve[:3]
    else:
        Currency = ''

    # xml params very useful for running all curves, but for ad hoc testing still useful to have direct input here:
    params = {'specific_curve': specific_curve,
                    'start_date': '2017-02-16',
                    'end_date': '2017-02-17',
                    'Currency': Currency,
                    'alpha': 0.5,
                    'gamma': 0.5,
                    'smfitCompressionShort': 0.1,
                    'smfitCompressionLong': 0.1,
                    'sigmaScaleLevels': 0.2,
                    'sigmaScaleChanges': 0.2,
                    'smfitSpldf': 4,
                    'write_data': False,
                    'debugplot': False,
                    'debugplot_curve':False,
                    'debugplot_ixlist': [-1],
                    'plot_f': False,
                    'sswidthLong': .05,
                    'sswidthShort': 0.8,
                    'maxIter': 10,
                    'numOutlierIterations': 3,
                    'numIterLevelGuess': 2,
                    'numBondsLevels': 200,
                    'numBondsChanges': 200,
                    'overwriteStart': False,
                    'fitSingle': fitSingle,
                    'debugplot_curve':True}

    xmlFileName = 'C:\ContentDev-MAC\macpy\utils\\xml_test.xml'
    xml_params = u.parse_xml(xmlFileName,params['specific_curve'])

    params.update(xml_params)

    if fitSingle:
        ssc = ngc.SmoothSplineCurve(**params)
        ssc.run()
    else:
        for c in currencyList:
            c_dict={'Currency':c}
            params.update(c_dict)
            ssc = ngc.SmoothSplineCurve(**params)
            ssc.run()

if __name__ == '__main__':
    main()