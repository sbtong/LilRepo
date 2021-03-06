import pandas as pd
import string
import datetime
import macpy.barclaysticker as bticker
import unittest



ticker = 'F5AN1613'

class Test_BarclaysTicker(unittest.TestCase):

    def test_parse_terms(self):
        actualTerms = bticker.parse_terms(ticker)
        expectedTerms = {"delay":"",
                "issuance_year": "",
                "descr_detail" : "",
            	"index_cd" : "",
            	"margin" : "",
            	"net_cpn" : "",
                "orig_wac":"",
            	"pmt_freq_cd":"",
                "pmt_lookback":"",
            	"pmt_reset_term":"",
            	"rate_life_cap":"",
            	"rate_life_floor":"",
                "rate_reset_cap":"",
                "rate_reset_flr":"",
            	"rate_reset_term":"",
            	"rate_teaser_term":"",
            	"wac":"",
            	"wala_term":"",
            	"wam_term":""}
        self.assertEqual(expectedTerms, actualTerms)
        

    def test_parse_delay(self):
        tickerParser = bticker.BarclaysHybridArmTicker(ticker)
        actualTerm = tickerParser.parse_delay(ticker)
        expectedTerm = 45 #delay is usually around 45 days
        self.assertEqual(expectedTerm, actualTerm)

    def test_parse_issuance_year(self):
        tickerParser = bticker.BarclaysHybridArmTicker(ticker)
        actualTerm = tickerParser.parse_issuance_year(ticker)
        expectedTerm = 2013
        self.assertEqual(expectedTerm, actualTerm)

    def test_parse_descr_detail(self):
        tickerParser = bticker.BarclaysHybridArmTicker(ticker)
        actualTerm = tickerParser.parse_descr_detail(ticker)
        expectedTerm = 'FN ARM'
        self.assertEqual(expectedTerm, actualTerm)

    def test_parse_index_cd(self):
        tickerParser = bticker.BarclaysHybridArmTicker(ticker)
        actualTerm = tickerParser.parse_index_cd(ticker)
        expectedTerm = 'WSJ1YLIB'
        self.assertEqual(expectedTerm, actualTerm)

    def test_parse_term_margin(self):
        tickerParser = bticker.BarclaysHybridArmTicker(ticker)
        actualTerm = tickerParser.parse_margin(ticker)
        expectedTerm = 0.50
        self.assertEqual(expectedTerm, actualTerm)

    def test_parse_term_net_cpn(self):
        tickerParser = bticker.BarclaysHybridArmTicker(ticker)
        actualTerm = tickerParser.parse_net_cpn(ticker)
        expectedTerm = 1.75
        self.assertEqual(expectedTerm, actualTerm)

    def test_parse_term_orig_wac(self):
        tickerParser = bticker.BarclaysHybridArmTicker(ticker)
        actualTerm = tickerParser.parse_orig_wac(ticker)
        expectedTerm = 1.75
        self.assertEqual(expectedTerm, actualTerm)

    def test_parse_term_pmt_freq_cd(self):
        tickerParser = bticker.BarclaysHybridArmTicker(ticker)
        actualTerm = tickerParser.parse_pmt_freq_cd(ticker)
        expectedTerm = 12
        self.assertEqual(expectedTerm, actualTerm)

    def test_parse_pmt_lookback(self):
        tickerParser = bticker.BarclaysHybridArmTicker(ticker)
        actualTerm = tickerParser.parse_pmt_lookback(ticker)
        expectedTerm = 45 #lookback is also around 45 days
        self.assertEqual(expectedTerm, actualTerm)

    def test_parse_pmt_reset_term(self):
        tickerParser = bticker.BarclaysHybridArmTicker(ticker)
        actualTerm = tickerParser.parse_pmt_reset_term(ticker)
        expectedTerm = 12
        self.assertEqual(expectedTerm, actualTerm)

    def test_parse_rate_teaser_term(self):
        tickerParser = bticker.BarclaysHybridArmTicker(ticker)
        actualTerm = tickerParser.parse_rate_teaser_term(ticker)
        expectedTerm = 120
        self.assertEqual(expectedTerm, actualTerm)

    def test_parse_wac(self):
        tickerParser = bticker.BarclaysHybridArmTicker(ticker)
        actualTerm = tickerParser.parse_wac(ticker)
        expectedTerm = 8.0
        self.assertEqual(expectedTerm, actualTerm)

    def test_parse_wala_term(self):
        tickerParser = bticker.BarclaysHybridArmTicker(ticker)
        actualTerm = tickerParser.parse_wala_term(ticker)
        expectedTerm = 10
        self.assertEqual(expectedTerm, actualTerm)

    def test_parse_wam_term(self):
        tickerParser = bticker.BarclaysHybridArmTicker(ticker)
        actualTerm = tickerParser.parse_wam_term(ticker)
        expectedTerm = 261
        self.assertEqual(expectedTerm, actualTerm)

    def test_parse_F5AN1613(self):
        actualTerms = bticker.parse_terms(ticker)
        expectedTerms = {
            "delay": 45,
            "issuance_year": 2013,
            "descr_detail" : "FN ARM",
            "index_cd" : "WSJ1YLIB",
            "margin" : 0.5,
            "net_cpn" : 1.75,
            "orig_wac": 2.25,
            "pmt_freq_cd": 12,
            "pmt_lookback": 45,
            "pmt_reset_term": 12,
            "rate_life_cap": 6.75,
            "rate_life_floor": 0.0,
            "rate_reset_cap": 2.0,
            "rate_reset_flr": 2.0,
            "rate_reset_term": 12,
            "rate_teaser_term": 60,
            "wac": 2.25,
            "wala_term": 0,
            "wam_term": 360
        }
        
        self.maxDiff = None
        self.assertEqual(expectedTerms, actualTerms)

class Test_BarclaysTickerDBWriter(unittest.TestCase):
    
    def test_db_writer(self):
        ticker = 'F5AN1613'
        btickerDb = bticker.BarclaysTickerDatabaseWriter(ticker)
        test = btickerDb.create_insert_sql()
        print test

    def test_populate_table(self):
        tickerSet = set(tickers.split())
        
        for x in tickerSet:
            print "Writing: " + x
            dbwriter = bticker.BarclaysTickerDatabaseWriter(x)
            dbwriter.write_to_db()



tickers = """
H5BN4002
H5BN4102
H5BN4202
H5BN4402
H5BN4502
H5BN4602
H5BN5002
H5BN5102
H5BN5202
H5BN5302
H5BN5502
H5BN5602
H5BN5702
H5BN3103
H5BN3203
H5BN3403
H5BN3503
H5BN4003
H5BN4103
H5BN4203
H5BN4403
H5BN4303
H5BN4603
H5BN4703
H5BN5003
H5BN5103
H5BN3104
H5BN3404
H5BN3604
H5BN3704
H5BN4004
H5BN4104
H5BN4204
H5BN4304
H5BN4404
H5BN4604
H5BN4504
H5BN5004
H5BN5104
H5BN4505
H5BN4405
H5BN4705
H5BN5005
H5BN5106
H5BN5406
H5BN5606
H5BN8106
H5BN5007
H5BN5407
H5BN5507
H5BN6307
H5BN7107
H5BN5008
H5BN5108
H5BN4708
H5BN5308
H5BN5408
H5BN6008
H5BN5109
H5BN3710
H7BN3510
H7BN3510
H7BN3610
H7BN3010
H7BN3110
H7BN3410
H7BN3410
H7BN2710
H7BN6008
H7BN6108
H7BN6608
H7BN6708
H7BN6508
H7BN3409
H7BN4109
H7BN5308
H7BN5408
H7BN5508
H7BN5608
H7BN4708
H7BN5208
H7BN5108
H7BN5308
H7BN5108
H7BN5008
H7BN4708
H7BN4408
H7BN4508
H7BN4508
H7BN4608
H7BN7407
H7BN7307
H7BN6707
H7BN6507
H7BN6107
H7BN6507
H7BN6207
H7BN5507
H7BN5607
H7BN5707
H7BN6107
H7BN6007
H7BN5607
H7BN5407
H7BN5307
H7BN4607
H7BN4707
H7BN7006
H7BN4307
H7BN6206
H7BN6306
H7BN6506
H7BN5506
H7BN5406
H7BN5306
H7BN5706
H7BN6006
H7BN5106
H7BN5206
H7BN4305
H7BN4506
H7BN5205
H7BN5605
H7BN5705
H7BN5005
H7BN5105
H7BN4705
H7BN4505
H7BN5504
H7BN5104
H7BN5004
H7BN5204
H7BN5304
H7BN4504
H7BN4704
H7BN4604
H7BN4404
H7BN4304
H7BN4204
H7BN4104
H7BN4004
H7BN5403
H7BN5103
H7BN5003
H7BN4703
H7BN5203
H7BN5303
H7BN5503
H7BN4603
H7BN4303
H7BN4403
H7BN4503
H7BN4203
H7BN4103
H7BN4003
H7BN3703
H7BN6202
H7BN6302
H7BN5702
H7BN5602
H7BN5502
H7BN5302
H7BN5402
H7BN5202
H7BN5102
H7BN5002
H7BN4602
H7BN4702
H7BN4502
H7BN4202
H7BN6301
H7BN6401
H7BN6601
H7BN5601
H7BN5701
H7BN6001
H7BN6101
H7BN6201
H7BN3210
H7BN2011
H7BN2111
H7BN2211
H7BN2311
H7BN2411
H7BN2511
H7BN2611
H7BN2711
H7BN3011
H7BN3111
H7BN3211
H7BN3311
H7BN3411
H7BN3511
H7BN3711
H7BN3611
H7BN1712
H7BN2012
H7BN2112
H7BN4311
H7BN2212
H7BN2312
H7BN2412
H7BN2512
H7BN2612
H7BN3212
H7BN2712
H7BN3012
H7BN3112
H7BN3112
H7BN3012
H7BN2712
H7BN3212
H7BN2612
H7BN2512
H7BN2412
H7BN2312
H7BN2212
H7BN2112
H7BN3611
H7BN4111
H7BN3711
H7BN3511
H7BN3411
H7BN3311
H7BN3211
H7BN3111
H7BN3011
H7BN2711
H7BN2611
H7BN2511
H7BN2411
H7BN2311
H7BN2211
H7BN4510
H7BN3210
H7BN4110
H7BN4210
H7BN4310
H7BN6201
H7BN6101
H7BN5701
H7BN5301
H7BN6701
H7BN4502
H7BN5002
H7BN5102
H7BN5402
H7BN5502
H7BN5602
H7BN5702
H7BN6402
H7BN6202
H7BN4203
H7BN4503
H7BN4603
H7BN5503
H7BN5603
H7BN6103
H7BN5303
H7BN5203
H7BN4703
H7BN5003
H7BN5103
H7BN5403
H7BN4204
H7BN4304
H7BN4404
H7BN4604
H7BN4704
H7BN4504
H7BN5304
H7BN5404
H7BN5204
H7BN5104
H7BN5504
H7BN5604
H7BN4505
H7BN4705
H7BN5105
H7BN5205
H7BN5305
H7BN4706
H7BN5206
H7BN6106
H7BN6006
H7BN5706
H7BN5706
H7BN5606
H7BN5306
H7BN5306
H7BN5406
H7BN5506
H7BN5606
H7BN5406
H7BN6706
H7BN6206
H7BN6106
H7BN7006
H7BN2707
H7BN4107
H7BN5007
H7BN5107
H7BN5307
H7BN5407
H7BN5607
H7BN5707
H7BN5407
H7BN5507
H7BN6007
H7BN6007
H7BN6107
H7BN6507
H7BN6607
H7BN6607
H7BN6107
H7BN6407
H7BN7307
H7BN4608
H7BN4608
H7BN4508
H7BN4408
H7BN5008
H7BN5108
H7BN5008
H7BN5308
H7BN5208
H7BN5108
H7BN4708
H7BN5608
H7BN5508
H7BN5308
H7BN5408
H7BN4109
H7BN3709
H7BN4009
H7BN3409
H7BN3509
H7BN3609
H7BN7408
H7BN6308
H7BN6408
H7BN6108
H7BN2710
H7BN6209
H7BN2510
H7BN5209
H7BN5509
H7BN5009
H7BN4309
H7BN4409
H7BN4509
H7BN4209
H7BN3410
H7BN3310
H7BN3110
H7BN3010
H7BN3610
H7BN3510
H7BN3710
H7BN3710
H7BN4010
HABN4010
HABN3710
HABN3510
HABN3610
HABN3010
HABN3110
HABN3310
HABN3410
HABN4209
HABN4509
HABN4409
HABN4309
HABN4709
HABN5109
HABN5609
HABN8708
HABN4009
HABN3709
HABN5408
HABN5408
HABN5508
HABN5608
HABN5208
HABN5108
HABN5208
HABN5308
HABN5008
HABN5108
HABN5008
HABN4708
HABN2507
HABN4308
HABN4508
HABN4608
HABN7307
HABN6507
HABN7007
HABN7007
HABN6707
HABN6407
HABN6107
HABN6607
HABN6507
HABN6107
HABN5707
HABN5607
HABN5507
HABN6007
HABN6207
HABN6207
HABN6307
HABN6307
HABN5507
HABN5407
HABN5707
HABN5607
HABN5407
HABN5207
HABN5207
HABN5307
HABN5107
HABN5007
HABN4607
HABN4407
HABN6106
HABN6206
HABN6506
HABN6306
HABN6406
HABN5406
HABN5606
HABN5506
HABN5406
HABN5306
HABN5606
HABN5706
HABN6006
HABN6106
HABN6006
HABN5206
HABN5106
HABN4706
HABN5006
HABN5405
HABN5205
HABN4405
HABN4505
HABN4605
HABN4605
HABN2605
HABN3705
HABN5504
HABN5004
HABN4604
HABN5203
HABN4203
HABN5502
HABN5301
HABN4310
HABN4210
HABN4110
HABN3210
HABN4610
HABN4410
HABN4209
HABN4710
HABN3011
HABN3111
HABN3211
HABN3311
HABN3411
HABN3511
HABN3711
HABN4011
HABN4111
HABN4211
HABN3611
HABN4411
HABN2512
HABN2612
HABN3212
HABN3312
HABN2712
HABN3012
HABN3112
H5BO4009
H5BO3310
H5BO3010
H7BO3510
H7BO5709
H7BO5309
H7BO4709
H7BO4409
H7BO6308
H7BO6108
H7BO5108
H7BO5607
H7BO5707
H7BO5507
H7BO6207
H7BO6607
H7BO6407
H7BO4608
H7BO4408
H7BO5008
H7BO4607
H7BO5307
H7BO5307
H7BO5207
H7BO5107
H7BO6206
H7BO3207
H7BO5205
H7BO6006
H7BO5606
H7BO5005
H7BO5008
H7BO4708
H7BO4408
H7BO4308
H7BO3208
H7BO4608
H7BO4508
H7BO6407
H7BO6607
H7BO6507
H7BO6207
H7BO6307
H7BO6007
H7BO5507
H7BO5407
H7BO5707
H7BO5607
H7BO5108
H7BO5308
H7BO5208
H7BO5608
H7BO5508
H7BO5408
H7BO5708
H7BO6408
H7BO4009
H7BO4109
H7BO3609
H7BO3509
H7BO3409
H7BO4409
H7BO4309
H7BO4509
H7BO4209
H7BO4609
H7BO5009
H7BO5109
H7BO3510
H7BO3610
H7BO3710
H7BO4010
H7BO3110
H7BO3310
H7BO3410
H7BO4110
HABO4210
HABO4310
HABO4410
HABO4610
HABO4010
HABO5109
HABO4709
HABO4609
HABO4509
HABO4309
HABO4409
HABO6508
HABO3709
HABO6308
HABO5708
HABO6008
HABO6108
HABO5408
HABO5508
HABO5608
HABO5208
HABO5308
HABO5108
HABO5607
HABO5707
HABO5407
HABO5507
HABO6007
HABO6307
HABO6207
HABO6507
HABO6607
HABO6407
HABO6107
HABO6707
HABO7007
HABO7307
HABO4508
HABO4608
HABO4308
HABO4708
HABO5008
HABO5105
HABO5606
HABO5706
HABO6006
HABO5306
HABO5506
HABO5406
HABO5305
HABO5505
HABO5006
HABO4606
HABO5106
HABO5206
HABO7306
HABO2207
HABO4407
HABO4507
HABO6206
HABO6106
HABO6406
HABO6306
HABO6506
HABO4406
HABO5107
HABO5007
HABO4607
HABO4707
HABO5207
HABO5307
H7AN4306
H7AN6105
H7AN5705
H7AN4703
H7AN3604
H7AN7507
H7AN6707
H7AN6107
H7AN6407
H7AN6507
H7AN6207
H7AN6307
H7AN6007
H7AN5507
H7AN5507
H7AN5707
H7AN5607
H7BN5507
H7BN5207
H7BN6007
H7BN6307
H7BN6107
H7BN7107
H7BN7207
H7BN6005
H7BN5505
H7BN5405
H7BN5506
H7BN6206
H7BN4107
H7AN5207
HABN5307
HABN4407
HABN6605
HABN6307
HABN6207
HABN5407
H7AO5407
H7AO5607
H7AO4107
H7AO5207
H7AO5007
H7BO4707
H7BO5307
H7BO6407
H7AO5707
HABO5707
HABO5607
HABO5407
HABO5207
HABO5507
HABO6007
HABO6107
HABO6607
HABO5307
HABO4707
HABO4607
HABO5107
HABO4107
HABO4507
HABO6106
HABO5106
HABO6006
F5BN2503
F5BN4003
F5BN3404
F5BN4004
F5BN4104
F5BN4005
F5BN5005
F5BN5205
F5BN5505
F5BN6005
F5BN6105
F5BN5506
F5BN6007
F5BN5008
F5BN5108
F5BN5508
F5BN4009
F5BN3210
F5BN3310
F5BN3510
F5BN2711
F5BN3011
F5BN3611
F7BN3611
F7BN3511
F7BN3311
F7BN3411
F7BN1512
F7BN1612
F7BN1712
F7BN2112
F7BN2212
F7BN2312
F7BN2412
F7BN2512
F7BN2612
F7BN2712
F7BN3012
F7BN3112
F7BN3212
F7BN2012
F7BN3312
F7BN3011
F7BN2711
F7BN3111
F7BN3211
F7BN2311
F7BN2411
F7BN2511
F7BN2611
F7BN3610
F7BN3710
F7BN1711
F7BN2011
F7BN2111
F7BN2211
F7BN3510
F7BN3410
F7BN3310
F7BN3210
F7BN2510
F7BN2610
F7BN2710
F7BN3010
F7BN3010
F7BN4209
F7BN4309
F7BN4409
F7BN4509
F7BN6508
F7BN5309
F7BN4009
F7BN3309
F7BN3509
F7BN5708
F7BN6008
F7BN6008
F7BN6208
F7BN6308
F7BN5408
F7BN5608
F7BN5008
F7BN5408
F7BN5608
F7BN5108
F7BN5208
F7BN5208
F7BN5308
F7BN4708
F7BN4508
F7BN4608
F7BN8307
F7BN8407
F7BN8707
F7BN3208
F7BN4208
F7BN4408
F7BN6307
F7BN7107
F7BN7407
F7BN7607
F7BN7707
F7BN8007
F7BN6007
F7BN6107
F7BN5607
F7BN5707
F7BN4707
F7BN5007
F7BN5107
F7BN5207
F7BN5307
F7BN5507
F7BN5506
F7BN5406
F7BN5606
F7BN6206
F7BN7006
F7BN3007
F7BN6005
F7BN5505
F7BN5006
F7BN5106
F7BN5206
F7BN5205
F7BN5405
F7BN5005
F7BN5105
F7BN4705
F7BN4105
F7BN4505
F7BN5604
F7BN3705
F7BN4404
F7BN3404
F7BN3604
F7BN3704
F7BN4003
F7BN5003
F7BN4103
F7BN4303
F7BN4504
F7BN4604
F7BN4704
F7BN5004
F7BN5204
F7BN4304
F7BN5404
F7BN5402
F7BN5502
F7BN4302
F7BN4402
F7BN5002
F7BN5202
F7BN5101
F7BN5201
F7BN5301
F7BN5601
F7BN5701
F7BN3202
F7BN6301
F7BN5601
F7BN5102
F7BN5002
F7BN5402
F7BN5602
F7BN5204
F7BN5104
F7BN5004
F7BN4704
F7BN4604
F7BN4504
F7BN5401
F7BN2604
F7BN4404
F7BN3705
F7BN4505
F7BN4605
F7BN4005
F7BN5205
F7BN5206
F7BN5006
F7BN4706
F7BN4506
F7BN4207
F7BN4205
F7BN6206
F7BN6306
F7BN6406
F7BN5606
F7BN5706
F7BN6006
F7BN5406
F7BN5306
F7BN5506
F7BN5507
F7BN5407
F7BN5307
F7BN5207
F7BN5107
F7BN5007
F7BN4707
F7BN5707
F7BN5607
F7BN6107
F7BN6207
F7BN6007
F7BN4607
F7BN6307
F7BN6507
F7BN4408
F7BN4208
F7BN3508
F7BN4608
F7BN4708
F7BN5008
F7BN5308
F7BN5208
F7BN5108
F7BN5608
F7BN5408
F7BN5508
F7BN6308
F7BN6208
F7BN6008
F7BN5708
F7BN3509
F7BN3609
F7BN3309
F7BN3409
F7BN4009
F7BN3709
F7BN2410
F7BN7008
F7BN4509
F7BN4409
F7BN4309
F7BN4209
F7BN4109
F7BN3110
F7BN3010
F7BN2710
F7BN2610
F7BN2510
F7BN3210
F7BN3310
F7BN3410
F7BN3510
F7BN2111
F7BN4210
F7BN4310
F7BN3710
F7BN3610
F7BN4010
F7BN4110
F7BN2611
F7BN2511
F7BN2411
F7BN2311
F7BN3211
F7BN3111
F7BN2711
F7BN3011
F7BN3212
F7BN3209
F7BN3112
F7BN3012
F7BN2712
F7BN2612
F7BN2512
F7BN2412
F7BN2312
F7BN2212
F7BN2112
F7BN3711
F7BN4011
F7BN3411
F7BN3311
F7BN3511
F7BN3611
FABN3611
FABN3511
FABN3311
FABN3411
FABN4011
FABN4111
FABN4211
FABN4311
FABN3711
FABN2412
FABN2512
FABN2612
FABN2712
FABN3012
FABN3112
FABN3212
FABN3312
FABN3011
FABN2711
FABN3111
FABN3211
FABN4110
FABN4010
FABN3610
FABN3710
FABN4310
FABN4410
FABN4210
FABN4610
FABN3510
FABN3410
FABN3310
FABN3210
FABN2710
FABN3110
FABN4109
FABN4209
FABN4309
FABN4409
FABN4509
FABN7108
FABN4609
FABN3709
FABN4009
FABN5708
FABN6008
FABN6208
FABN6108
FABN5508
FABN5408
FABN5608
FABN5108
FABN5208
FABN5308
FABN5008
FABN4708
FABN4608
FABN4508
FABN4008
FABN4208
FABN4308
FABN4408
FABN6507
FABN6607
FABN6307
FABN6407
FABN7707
FABN7107
FABN7007
FABN4107
FABN6007
FABN6207
FABN6107
FABN5607
FABN5707
FABN5107
FABN5207
FABN5307
FABN5407
FABN5507
FABN5506
FABN5306
FABN5406
FABN6006
FABN5706
FABN5606
FABN6406
FABN6306
FABN6206
FABN6106
FABN6706
FABN5505
FABN5705
FABN4706
FABN5006
FABN5206
FABN5106
FABN5205
FABN5405
FABN4705
FABN5105
FABN5005
FABN3605
FABN4204
FABN4503
FABN4603
FABN4103
FABN4203
FABN5203
FABN4703
FABN4504
FABN4604
FABN4704
FABN5004
FABN5104
FABN5204
FABN5504
FABN5304
FABN5702
FABN5402
FABN5502
FABN5002
FABN5102
FABN5301
F5BO2311
F7BO2311
F7BO2411
F7BO2511
F7BO2611
F7BO3211
F7BO3111
F7BO2711
F7BO3011
F7BO2211
F7BO2111
F7BO3610
F7BO2410
F7BO4309
F7BO4109
F7BO3110
F7BO3010
F7BO2610
F7BO3210
F7BO3310
F7BO3410
F7BO2012
F7BO2512
F7BO2412
F7BO2212
F7BO2312
F7BO2112
F7BO1712
F7BO3411
F7BO3311
F7BO5507
F7BO5207
F7BO5107
F7BO4507
F7BO5707
F7BO6107
F7BO6207
F7BO7507
F7BO6707
F7BO6507
F7BO4308
F7BO5008
F7BO5308
F7BO5108
F7BO5608
F7BO5408
F7BO5508
F7BO3709
F7BO3709
F7BO4009
F7BO3609
F7BO6108
F7BO6008
F7BO5708
F7BO5408
F7BO5608
F7BO5208
F7BO5308
F7BO5008
F7BO4708
F7BO4508
F7BO4608
F7BO4408
F7BO7507
F7BO7407
F7BO7307
F7BO6207
F7BO6107
F7BO6007
F7BO5707
F7BO5607
F7BO4507
F7BO4707
F7BO5107
F7BO5007
F7BO5207
F7BO5307
F7BO5507
F7BO5407
F7BO6706
F7BO4307
F7BO4407
F7BO6106
F7BO5606
F7BO6006
F7BO5406
F7BO5306
F7BO5506
F7BO3205
F7BO4505
F7BO5106
F7BO4606
F7BO4306
F7BO4406
F7BO4106
F7BO3311
F7BO3411
F7BO3511
F7BO3611
F7BO2112
F7BO3711
F7BO2312
F7BO2212
F7BO2412
F7BO2512
F7BO3112
F7BO3012
F7BO2712
F7BO2612
F7BO2012
F7BO3412
F7BO3410
F7BO3510
F7BO3310
F7BO3210
F7BO2510
F7BO2710
F7BO3010
F7BO3110
F7BO4109
F7BO4209
F7BO4309
F7BO4409
F7BO5609
F7BO4709
F7BO5109
F7BO4609
F7BO4509
F7BO3610
F7BO3710
F7BO4010
F7BO4110
F7BO4210
F7BO3011
F7BO2711
F7BO3111
F7BO3211
F7BO2611
F7BO2511
F7BO2411
F7BO2311
FABO2611
FABO3211
FABO3111
FABO2711
FABO3011
FABO4210
FABO4410
FABO4510
FABO4310
FABO4710
FABO5310
FABO4110
FABO4010
FABO3710
FABO3610
FABO4509
FABO4609
FABO5109
FABO4709
FABO5009
FABO4409
FABO4309
FABO4209
FABO4109
FABO3210
FABO3310
FABO3510
FABO3410
FABO3412
FABO3512
FABO3312
FABO3212
FABO2612
FABO2712
FABO3012
FABO3112
FABO2512
FABO2412
FABO3711
FABO4011
FABO3611
FABO3511
FABO3411
FABO3311
FABO4106
FABO4506
FABO6005
FABO4606
FABO4706
FABO5006
FABO5106
FABO5206
FABO4105
FABO3505
FABO4705
FABO5305
FABO5304
FABO4304
FABO5504
FABO5404
FABO5204
FABO4604
FABO4504
FABO4404
FABO5506
FABO5306
FABO5406
FABO6006
FABO5606
FABO5706
FABO6106
FABO6206
FABO6306
FABO6406
FABO4407
FABO4307
FABO4007
FABO7206
FABO6706
FABO6606
FABO7006
FABO7106
FABO5407
FABO5507
FABO5307
FABO5207
FABO5007
FABO5107
FABO4707
FABO4507
FABO5607
FABO5707
FABO6007
FABO6107
FABO6207
FABO7107
FABO6507
FABO6707
FABO6607
FABO6407
FABO6307
FABO4408
FABO4008
FABO4108
FABO4608
FABO4508
FABO4708
FABO5008
FABO5308
FABO5208
FABO5108
FABO5608
FABO5408
FABO5508
FABO5708
FABO6008
FABO6108
FABO3609
FABO3409
FABO4009
FABO3709
F3AN3302
F3AN5202
F3AN6502
F5AN4502
F5AN5002
F5AN4404
F5AN4004
F5AN5305
F5AN5105
F5AN9700
F5BN5307
F7AN7107
F7AN5508
F7AN5608
F7AN5308
F7AN5307
F7AN5407
F7AN6306
F7AN5606
F7AN3602
F7AN4002
F7AN4202
F7AN5002
F7AN4602
F7AN4702
F7AN5302
F7AN7002
F7AN6302
F7AN3403
F7AN3603
F7AN5702
F7AN5602
F7AN5405
F7AN5205
F7AN6604
F7AN4605
F7AN6405
F7AN4004
F7AN4204
F7AN4404
F7AN3504
F7AN3704
F7AN3604
F7AN4703
F7AN4003
F7AN4203
F7AN5004
F7BN4704
F7BN5705
F7BN5505
F7BN5704
F7BN4705
F7BN5202
F7BN4402
F7BN4302
F7BN4202
F7BN5507
F7BN7307
F7BN6607
F7BN5403
F7AN4504
F7AN5104
F7AN6002
F7AN3303
F7AN6407
F7AN5407
F7BN5102
F7BN8000
F7BN7700
FAAN5302
FAAN6302
FAAN5502
FAAN5104
FAAN5504
FAAN5304
FAAN4604
FAAN5003
FAAN5305
FAAN5607
FAAN5406
FABN5606
FABN6006
FABN7106
FABN7406
FABN6107
FABN4605
FABN7700
FABN7299
FABN7599
FABN7799
FABN7600
FABN8000
FABN8200
F7BO6307
F7BO6707
F7BO5707
F7BO6007
F7BO6207
F7AO6107
F7AO5507
F7AO8207
F7BO5707
FAAO5607
FAAO6006
FAAO5606
FAAO5406
FAAO5306
FAAO5506
FAAO4106
FAAO5106
FAAO4603
FABO5106
FABO5205
FABO5506
FABO5306
FABO5706
FABO6006
FABO5606
FABO6406
FABO6506
FABO6206
FABO6106
FABO5607
FABO5507
FABO4507
FABO6307
FABO6107
G5AN3012
G7AN3012
G7AN3411
G7AN4011
G7AN2412
G7AN2411
G7AN3011
G7BN3011
G7BN2411
G7BN4409
G7BN5009
G7BN2412
G7BN4011
G7BN3411
G7BN3012"""



        
if __name__ == '__main__':
    unittest.main()
