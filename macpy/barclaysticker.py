from string import Template
from datetime import datetime
import macpy.utils.database as db

class IndexStructure:
    def __init__(self, initialCap, periodicCap, lifetimeCap, indexCode, resetTermInMonths):
        self.initialCap = initialCap
        self.periodicCap = periodicCap
        self.lifeTime = lifetimeCap
        self.indexCode = indexCode
        self.resetTermInMonths = resetTermInMonths

def parse_terms(ticker):

    tickerParser = BarclaysHybridArmTicker(ticker)
    terms = tickerParser.parse_terms()
    return terms


class BarclaysHybridArmTicker:

    def __init__(self, ticker):

        characterCount = len(ticker)
        if characterCount != 8:
            raise ValueError(ticker + " does not have 8 characters. Ticker is not in the proper format")
        
        self.ticker = ticker
        self.parseMap = {
                    "TickerName": self.get_ticker,
                    "delay": self.parse_delay,
                    "descr_detail" : self.parse_descr_detail,
                    "issuance_year": self.parse_issuance_year,
            	    "index_cd" : self.parse_index_cd,
            	    "margin" : self.parse_margin,
            	    "net_cpn" : self.parse_net_cpn,
                    "orig_wac": self.parse_orig_wac,
            	    "pmt_freq_cd": self.parse_pmt_freq_cd,
                    "pmt_lookback": self.parse_pmt_lookback,
            	    "pmt_reset_term": self.parse_pmt_reset_term,
            	    "rate_life_cap": self.parse_rate_life_cap,
            	    "rate_life_floor": self.parse_rate_life_floor,
                    "rate_reset_cap": self.parse_rate_reset_cap,
                    "rate_reset_flr": self.parse_rate_reset_flr,
            	    "rate_reset_term": self.parse_rate_reset_term,
            	    "rate_teaser_term": self.parse_rate_teaser_term,
            	    "wac": self.parse_wac,
            	    "wala_term": self.parse_wala_term,
            	    "wam_term": self.parse_wam_term}

        self.indexStructureMap = {
            "A":IndexStructure(2.0, 2.0, 5.0, 'WSJ1YLIB', 12),
            "B":IndexStructure(5.0, 2.0, 5.0, 'WSJ1YLIB', 12),
            "C":IndexStructure(2.0, 2.0, 5.0, 'WSJ1YLIB', 12),
            "D":IndexStructure(5.0, 2.0, 5.0, 'WSJ1YLIB', 12),
            "E":IndexStructure(2.0, 2.0, 6.0, 'WSJ1YLIB', 12),
            "M":IndexStructure(1.0, 1.0, 5.0, 'WSJ1YLIB', 12),
            "N":IndexStructure(1.0, 1.0, 5.0, 'WSJ1YLIB', 12)
            }

        self.hybridTermInMonths = {
            "1":12,
            "3":36,
            "5":60,
            "7":80,
            "A":120
        }

        self.agencyCode = {
        "H":"FN ARM", 
        "G":"FN ARM",
        "F":"FN ARM"
        }

    def get_ticker(self):

        return self.ticker

    def parse_terms(self):
        termsMap = { k : v() for (k,v) in self.parseMap.iteritems() }
        return termsMap

    def parse_delay(self):
        term = 45 #use same delay term
        return term

    def parse_issuance_year(self):
        #ticker example: F5AN1613  ( 13 => 2013 ) issuance year
        term = self.ticker[-2:]
        term = int("20" + term)
        return term

    def parse_descr_detail(self):
        #ticker example: F5AN1613  ( F => Agency ) agency cd
        term = self.ticker[0]
        term = 'FN ARM' #EJV Arm Pool Type

        return term

    def parse_index_cd(self):
        term = 'WSJ1YLIB'
        return term

    def parse_margin(self):
        term = 0.50
        return term

    def parse_net_cpn(self):
        #ticker example: F5AN1613  ( 16 => coupon ) 1 + 6.0/8.0
        couponFirstPart = int(self.ticker[4:5])
        couponSecondPart = float(self.ticker[5:6])/8.0
        coupon = couponFirstPart + couponSecondPart
        term = float("{0:.2f}".format(coupon))
        return term

    def parse_orig_wac(self):
        term = self.parse_net_cpn() + self.parse_margin()
        return term

    def parse_pmt_freq_cd(self):
        term = 12
        return term

    def parse_pmt_lookback(self):
        term = 45 #days
        return term

    def parse_pmt_reset_term(self):
        term = 12
        return term

    def parse_rate_life_cap(self):
        index_structure = self.parse_index_structure()
        term = self.indexStructureMap[index_structure].lifeTime + self.parse_net_cpn()
        return term

    def parse_index_structure(self):
        #ticker example: F5AN1613  ( A => index structure ) third character
        index_structure = self.ticker[2]
        return index_structure

    def parse_rate_life_floor(self):
        term = 0.0
        return term

    def parse_rate_reset_cap(self):
        index_structure = self.parse_index_structure()
        term = self.indexStructureMap[index_structure].periodicCap
        return term

    def parse_rate_reset_flr(self):
        index_structure = self.parse_index_structure()
        term = self.indexStructureMap[index_structure].periodicCap
        return term

    def parse_rate_reset_term(self):
        term = 12 #resest term in months
        return term

    def parse_rate_teaser_term(self):
        #ticker example: F5AN1613  ( 5 => hybrid term in months ) second character
        hybridPeriod = self.ticker[1]
        term = self.hybridTermInMonths[hybridPeriod]
        return term

    def parse_wac(self):
        term = self.parse_net_cpn() + self.parse_margin()
        return term    

    def parse_wala_term(self):
        term = 0
        return term

    def parse_wam_term(self):
        term = 360
        return term

class BarclaysTickerDatabaseWriter(object):
    def __init__(self, ticker):
        self.ticker = ticker

    def create_insert_sql(self):
        terms = parse_terms(self.ticker)
        sqlTemplate = Template(
"""INSERT INTO [MarketData].[dbo].[BarclaysHybridArmTickerTerms]
       ([TickerName]
      ,[delay]
      ,[descr_detail]
      ,[index_cd]
      ,[issuance_year]
      ,[margin]
      ,[net_cpn]
      ,[orig_wac]
      ,[pmt_freq_cd]
      ,[pmt_lookback]
      ,[pmt_reset_term]
      ,[rate_life_cap]
      ,[rate_life_floor]
      ,[rate_reset_cap]
      ,[rate_reset_flr]
      ,[rate_reset_term]
      ,[rate_teaser_term]
      ,[wac]
      ,[wala_term]
      ,[wam_term]
      ,[Lud]
      ,[Lub])
  Values 
       ('$TickerName'
      ,$delay
      ,'$descr_detail'
      ,'$index_cd'
      ,$issuance_year
      ,$margin
      ,$net_cpn
      ,$orig_wac
      ,'$pmt_freq_cd'
      ,$pmt_lookback
      ,$pmt_reset_term
      ,$rate_life_cap
      ,$rate_life_floor
      ,$rate_reset_cap
      ,$rate_reset_flr
      ,$rate_reset_term
      ,$rate_teaser_term
      ,$wac
      ,$wala_term
      ,$wam_term
      ,'$Lud'
      ,'$Lub')""")

        insertSql = sqlTemplate.substitute(Lud=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), Lub='wgajate', **terms)

        return insertSql

    def write_to_db(self):
        insertSql = self.create_insert_sql()
        db.MSSQL.execute_commit(insertSql)
        

