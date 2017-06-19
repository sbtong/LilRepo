import unittest
import macpy.extraction as extraction


class Test_Extraction(unittest.TestCase):

    def curvenodequote(self):
        args = {
        'CurveShortName' : 'US.USD.GVT.ZC',
        'StartDate' : '2014-01-01',
        'EndDate' : '2014-01-15',
        'InterimTable' : 0}
    
        cnq = extraction.CurveNodeQuote()
        sqlstatement = cnq.create_sql(**args)
        self.__validate_sql(sqlstatement)
        df = cnq.extract_dataframe(**args)
        self.__dataframe_is_not_empty(df)
    
    
    def test_derivcurvefilteredbonds(self):
        args = {
        'CurveShortName' : 'CA.CAD.GVT.ZC',
        'StartDate' : '2014-01-01',
        'EndDate' : '2014-02-01'}
    
        fb = extraction.DerivCurveFilteredBonds()
        sqlstatement = fb.create_sql(**args)
        self.__validate_sql(sqlstatement)

        df = fb.extract_dataframe(**args)
        self.__dataframe_is_not_empty(df)

  
    def __validate_sql(self, sqlstatement):    
        assert sqlstatement is not None
        assert '$' not in sqlstatement
    
    def __dataframe_is_not_empty(self, df):    
        assert len(df) > 0 

if __name__ == '__main__':
    unittest.main()