import unittest
import macpy.utils.database as db


class Test_Simple_Query(unittest.TestCase):
    def test_simple_query(self):
        sqlquery = 'select top 1 * from MarketData.dbo.Curve'
        df = db.MSSQL.extract_dataframe(sqlquery)
        assert len(df) > 0 

        
if __name__ == '__main__':
    unittest.main()