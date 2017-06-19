import pandas as pd
import numpy as np
import macpy.utils.database as db
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

class Compare(object):
    def __init__(self, tradeDate, currency, enviroment=None, config=None):
        self.tradeDate=tradeDate
        self.currency=currency
        self.enviroment = enviroment
        self.config = config

    def run(self):
        curve_name = db.convert_currency_to_curveName(self.currency)
        csharp = db.DerivedCurveFilteredBond(self.tradeDate, curve_name, self.enviroment, self.config).extract_from_db()
        csharp_instr = sorted(set(list(csharp['InstrCode'])))

        python = db.DerivedCurveFilteredBondResearch(self.currency, self.tradeDate, self.enviroment, self.config).extract_from_db()
        python_instr = sorted(set(list(python['InstrCode'])))

        diff = set(csharp_instr)^set(python_instr)

        print 'Legacy filtered bond instrument id\'s:'
        print csharp_instr
        print '\n'

        print 'New filtered bond instrument id\'s:'
        print python_instr
        print '\n'

        print 'Symmetric difference:'
        print diff

        if len(diff) > 0:
            return 1
        else:
            return 0

if __name__ == '__main__':
    Compare('2016-04-15', 'JPY', 'PROD').run()