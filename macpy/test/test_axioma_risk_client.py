import unittest
import macpy.AxiomaRiskClient as riskClient


class AxiomaRisk(unittest.TestCase):

    def axioma_risk_client(self):
        runner = riskClient.RiskJobRunner()
        #dataIds = ['D7KBAUV7M9']
        #dataIds = ['DRM8T7MA55']
        #dataIds = ['D7KBAUV7M9']
        dataIds = ['D7KBAUV7M9']
    
        res = runner.run(datetime.date(2014, 5, 30), dataIds)    


if __name__ == '__main__':
    unittest.main()