import macpy.bond as bond
import macpy.bondPortfolio as bondPortfolio
import macpy.next_gen_curve as ngc
import pandas as pd
import os
import string
import datetime
import macpy.utils.database as db
import numpy as np
import unittest
import macpy.utils.ngc_utils as u
dir_path = os.path.dirname(os.path.realpath(__file__))
import macpy.utils.ngc_queries as q
import plotly

class Test_NextGenFit(unittest.TestCase):

    def test_USDJPMSEN_20170217(self):
        params = {'specific_curve': 'USD-JPM-SEN',
                  'start_date': '2017-02-16',
                  'end_date': '2017-02-17',
                  'Currency': 'USD',
                  'alpha': 0.5,
                  'gamma': 0.5,
                  'smfitCompressionShort': 0.1,
                  'smfitCompressionLong': 0.1,
                  'sigmaScaleLevels': 0.2,
                  'sigmaScaleChanges': 0.2,
                  'smfitSpldf': 4,
                  'write_data': False,
                  'debugplot': False,
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
                  'fitSingle': True,
                  'IndustryGroup':'BANKS',
                  'Sector':'FINS',
                  'SectorGroup':'FIN',
                  'Market':'CR',
                  'Region':'NAMR',
                  'RegionGroup':'DVL',
                  'PricingTier':'SEN',
                  'RiskEntityId':100054757}

        ssc = ngc.SmoothSplineCurve(**params)
        s_prev = pd.read_csv(dir_path + '/data/test_ngc_USD_20170216.csv')
        s_prev['MatDate'] = s_prev.MatDate.apply(lambda x: pd.to_datetime(x))
        s_prev = u.process_data(s_prev, ssc.curveCol, ssc.spreadCol, ssc.weightCol, ssc.tenorCol)
        s_i= pd.read_csv(dir_path + '/data/test_ngc_USD_20170217.csv')
        s_i['MatDate'] = s_i.MatDate.apply(lambda x: pd.to_datetime(x))
        s_i = u.process_data(s_i, ssc.curveCol, ssc.spreadCol, ssc.weightCol, ssc.tenorCol)
        res_f, res_d = ssc.fit_date_pair(ssc.end_date, s_i, s_prev)

        level = np.array((res_f['yout']))
        change = np.array((res_d['yout']))
        coupon = np.array(res_f['coupon'])
        AmtOut = np.array(res_f['AmtOutstanding_pdf'])
        sumAmtOut = res_f['SumAmtOutstanding']
        NumOutliersLvls = res_f['IssuerBonds'].loc[res_f['IssuerBonds'].outlier == True].shape[0]
        NumOutliersChgs = res_d['IssuerBonds'].loc[res_d['IssuerBonds'].outlier == True].shape[0]

        level_benchmark = [-0.196808916,-0.189155974,-0.180262015,-0.169851257,-0.126898261,-0.064691967,0.013955708,
                           0.102439586,0.191117972,0.345704433,0.465388727,0.563873621,0.651193858,0.731657507,
                           0.87632621,1.106199319,1.344302567,1.462078878,1.604637737,1.791209254,1.924716031,
                           1.997259171,2.116938316]

        change_benchmark = [0.004741938,0.004710484,0.004674213,0.004631991,0.004460269,0.004212776,0.003890711,
                            0.003488707,0.002997342,0.001718407,0.000126605,-0.001623914,-0.003377854,-0.005014715,
                            -0.007670216,-0.010143344,-0.009309722,-0.007775583,-0.005474535,-0.00233817,-9.55678E-05,
                            0.001122696,0.003132543]

        coupon_benchmark = [4.803445937,4.803445937,4.803445937,4.803445937,4.803445937,4.803445937,4.726620933,
                            4.649994406,4.573883244,4.424550201,4.281198578,4.146269844,4.021646887,3.908201134,
                            3.715644631,3.465927418,3.399570583,3.522997717,3.886265582,4.738699523,5.569818904,
                            5.569818904,5.569818904]

        AmtOut_benchmark = [418.5196862,1089.487092,2336.801063,4286.66302,13321.34517,20661.67066,21271.22328,
                            18635.04884,17761.7212,25217.86496,38180.08031,49455.82375,55189.73523,55720.32649,
                            51353.56203,44806.82847,22323.22148,16786.12668,14384.49087,3957.960959,449.8145562,
                            33.97288135,0.125430148]

        sumAmtOut_benchmark = 82164.927

        NumOutliersLvls_benchmark = 13
        NumOutliersChgs_benchmark = 16

        for i in range(0,len(ssc.tenorGrid)):
            self.assertAlmostEquals(level[i], level_benchmark[i], delta=1e-5)
            self.assertAlmostEquals(change[i], change_benchmark[i], delta=1e-5)
            self.assertAlmostEquals(coupon[i], coupon_benchmark[i], delta=1e-5)
            self.assertAlmostEquals(AmtOut[i], AmtOut_benchmark[i], delta=1e-2)

        self.assertAlmostEquals(sumAmtOut, sumAmtOut_benchmark, delta=1e-2)
        self.assertEquals(NumOutliersLvls, NumOutliersLvls_benchmark)
        self.assertEquals(NumOutliersChgs, NumOutliersChgs_benchmark)

    def test_BRLCGASBZSEN_20170217(self):
        params = {'specific_curve': 'BRL-CGASBZ-SEN',
                  'start_date': '2017-02-16',
                  'end_date': '2017-02-17',
                  'Currency': 'BRL',
                  'alpha': 0.5,
                  'gamma': 0.5,
                  'smfitCompressionShort': 0.1,
                  'smfitCompressionLong': 0.1,
                  'sigmaScaleLevels': 0.2,
                  'sigmaScaleChanges': 0.2,
                  'smfitSpldf': 4,
                  'write_data': False,
                  'debugplot': False,
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
                  'fitSingle': True,
                  'IndustryGroup':'ENRGY',
                  'Sector':'ENRG',
                  'SectorGroup':'NFN',
                  'Market':'CR',
                  'Region':'LATM',
                  'RegionGroup':'EMG',
                  'PricingTier':'SEN',
                  'RiskEntityId':100089156}

        ssc = ngc.SmoothSplineCurve(**params)
        s_prev = pd.read_csv(dir_path + '/data/test_ngc_BRL_20170216.csv')
        s_prev['MatDate'] = s_prev.MatDate.apply(lambda x: pd.to_datetime(x))
        s_prev = u.process_data(s_prev, ssc.curveCol, ssc.spreadCol, ssc.weightCol, ssc.tenorCol)
        s_i= pd.read_csv(dir_path + '/data/test_ngc_BRL_20170217.csv')
        s_i['MatDate'] = s_i.MatDate.apply(lambda x: pd.to_datetime(x))
        s_i = u.process_data(s_i, ssc.curveCol, ssc.spreadCol, ssc.weightCol, ssc.tenorCol)
        res_f, res_d = ssc.fit_date_pair(ssc.end_date, s_i, s_prev)

        level = np.array((res_f['yout']))
        change = np.array((res_d['yout']))
        coupon = np.array(res_f['coupon'])
        AmtOut = np.array(res_f['AmtOutstanding_pdf'])
        sumAmtOut = res_f['SumAmtOutstanding']
        NumOutliersLvls = res_f['IssuerBonds'].loc[res_f['IssuerBonds'].outlier == True].shape[0]
        NumOutliersChgs = res_d['IssuerBonds'].loc[res_d['IssuerBonds'].outlier == True].shape[0]

        level_benchmark = [3.639270009,3.635581858,3.631229174,3.626112279,3.604614673,3.573592653,3.538688393,
                           3.508285083,3.48681827,3.460822355,3.444348142,3.431038716,3.417569738,3.402996145,
                           3.371738688,3.315298591,3.291832387,3.287398857,3.281785588,3.27444077,3.268675277,
                           3.26391827,3.256383171]

        change_benchmark = [-0.013464771,-0.013289629,-0.013082931,-0.012839941,-0.01181907,-0.010316223,-0.008191955,
                            -0.005673583,-0.003344836,-4.84418E-05,0.002611058,0.005341764,0.007332062,0.008157101,
                            0.007848115,0.005027588,0.003696972,0.00344508,0.003126162,0.002708864,0.002381297,
                            0.002111027,0.001682919]

        coupon_benchmark = [9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5,9.5]

        AmtOut_benchmark = [0.80119666,2.957569652,8.591325671,20.4937953,114.982401,258.4269135,322.4475716,
                            270.7397857,172.5916645,41.10762252,6.390417987,0.788363977,0.085983293,0.008830086,
                            8.92814E-05,1.17096E-08,5.28673E-14,3.01182E-17,1.05659E-21,3.03092E-28,6.03089E-34,
                            5.16696E-39,9.2791E-48]

        sumAmtOut_benchmark = 163.462

        NumOutliersLvls_benchmark = 0
        NumOutliersChgs_benchmark = 0

        for i in range(0,len(ssc.tenorGrid)):
            self.assertAlmostEquals(level[i], level_benchmark[i], delta=1e-5)
            self.assertAlmostEquals(change[i], change_benchmark[i], delta=1e-5)
            self.assertAlmostEquals(coupon[i], coupon_benchmark[i], delta=1e-5)
            self.assertAlmostEquals(AmtOut[i], AmtOut_benchmark[i], delta=1e-2)

        self.assertAlmostEquals(sumAmtOut, sumAmtOut_benchmark, delta=1e-2)
        self.assertEquals(NumOutliersLvls, NumOutliersLvls_benchmark)
        self.assertEquals(NumOutliersChgs, NumOutliersChgs_benchmark)

    def test_USDMEXSOV_20170217(self):
        params = {'specific_curve': 'USD-MEX-SOV',
                  'start_date': '2017-02-16',
                  'end_date': '2017-02-17',
                  'Currency': 'USD',
                  'alpha': 0.5,
                  'gamma': 0.5,
                  'smfitCompressionShort': 0.1,
                  'smfitCompressionLong': 0.1,
                  'sigmaScaleLevels': 0.2,
                  'sigmaScaleChanges': 0.2,
                  'smfitSpldf': 4,
                  'write_data': False,
                  'debugplot': False,
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
                  'fitSingle': True,
                  'IndustryGroup':'SOVRN',
                  'Sector':'SOVR',
                  'SectorGroup':'SOV',
                  'Market':'SV',
                  'Region':'LATM',
                  'RegionGroup':'EMG',
                  'PricingTier':'SOV',
                  'RiskEntityId':100087207}

        ssc = ngc.SmoothSplineCurve(**params)
        s_prev = pd.read_csv(dir_path + '/data/test_ngc_USD_20170216.csv')
        s_prev['MatDate'] = s_prev.MatDate.apply(lambda x: pd.to_datetime(x))
        s_prev = u.process_data(s_prev, ssc.curveCol, ssc.spreadCol, ssc.weightCol, ssc.tenorCol)
        s_i= pd.read_csv(dir_path + '/data/test_ngc_USD_20170217.csv')
        s_i['MatDate'] = s_i.MatDate.apply(lambda x: pd.to_datetime(x))
        s_i = u.process_data(s_i, ssc.curveCol, ssc.spreadCol, ssc.weightCol, ssc.tenorCol)
        res_f, res_d = ssc.fit_date_pair(ssc.end_date, s_i, s_prev)

        level = np.array((res_f['yout']))
        change = np.array((res_d['yout']))
        coupon = np.array(res_f['coupon'])
        AmtOut = np.array(res_f['AmtOutstanding_pdf'])
        sumAmtOut = res_f['SumAmtOutstanding']
        NumOutliersLvls = res_f['IssuerBonds'].loc[res_f['IssuerBonds'].outlier == True].shape[0]
        NumOutliersChgs = res_d['IssuerBonds'].loc[res_d['IssuerBonds'].outlier == True].shape[0]

        level_benchmark = [-0.098527594,-0.086182075,-0.072525527,-0.057339445,0.00012731,0.07911658,0.178537599,
                           0.291659562,0.40969602,0.63125684,0.814513927,0.962500785,1.083644245,1.183788535,
                           1.340099897,1.562296293,1.816560052,1.928189624,2.045156092,2.094607374,2.129059189,
                           2.157556369,2.202800959]

        change_benchmark = [0.002885665,0.00293715,0.002994558,0.003060622,0.003321236,0.003690531,0.004158029,
                            0.004694534,0.005265118,0.006423753,0.007465926,0.00824797,0.008862652,0.009394525,
                            0.010188114,0.010477126,0.009239139,0.008202358,0.006548576,0.005778133,0.005241252,
                            0.004797166,0.004092097]

        coupon_benchmark = [4.210119439,4.210119439,4.210119439,4.210119439,4.210119439,4.210119439,4.210119439,
                            4.210119439,4.210119439,4.210119439,4.210119439,4.210119439,4.210119439,4.210119439,
                            4.278942815,4.416039056,4.617410218,4.743714607,4.911659958,5.106883211,5.195913038,
                            5.229032345,5.289789923]

        AmtOut_benchmark = [2.03779E-10,4.16166E-09,5.89354E-08,6.14644E-07,0.000170663,0.010458807,0.230912319,
                            2.477029684,15.68216842,211.3629411,1105.07218,3227.814973,6514.899936,10358.20745,
                            17276.75142,20521.89757,19404.59498,30654.52833,36399.46447,11921.26235,1499.126523,
                            120.0210026,0.463794261]

        sumAmtOut_benchmark = 35494.078

        NumOutliersLvls_benchmark = 4
        NumOutliersChgs_benchmark = 2

        for i in range(0,len(ssc.tenorGrid)):
            self.assertAlmostEquals(level[i], level_benchmark[i], delta=1e-5)
            self.assertAlmostEquals(change[i], change_benchmark[i], delta=1e-5)
            self.assertAlmostEquals(coupon[i], coupon_benchmark[i], delta=1e-5)
            self.assertAlmostEquals(AmtOut[i], AmtOut_benchmark[i], delta=1e-2)

        self.assertAlmostEquals(sumAmtOut, sumAmtOut_benchmark, delta=1e-2)
        self.assertEquals(NumOutliersLvls, NumOutliersLvls_benchmark)
        self.assertEquals(NumOutliersChgs, NumOutliersChgs_benchmark)

    def test_USDVENZSOV_20170217(self):
        params = {'specific_curve': 'USD-VENZ-SOV',
                  'start_date': '2017-02-16',
                  'end_date': '2017-02-17',
                  'Currency': 'USD',
                  'alpha': 0.5,
                  'gamma': 0.5,
                  'smfitCompressionShort': 0.1,
                  'smfitCompressionLong': 0.1,
                  'sigmaScaleLevels': 0.2,
                  'sigmaScaleChanges': 0.2,
                  'smfitSpldf': 4,
                  'write_data': False,
                  'debugplot': False,
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
                  'fitSingle': True,
                  'IndustryGroup': 'SOVRN',
                  'Sector': 'SOVR',
                  'SectorGroup': 'SOV',
                  'Market': 'SV',
                  'Region': 'LATM',
                  'RegionGroup': 'EMG',
                  'PricingTier': 'SOV',
                  'RiskEntityId': 100087207}

        ssc = ngc.SmoothSplineCurve(**params)
        s_prev = pd.read_csv(dir_path + '/data/test_ngc_USD_20170216.csv')
        s_prev['MatDate'] = s_prev.MatDate.apply(lambda x: pd.to_datetime(x))
        s_prev = u.process_data(s_prev, ssc.curveCol, ssc.spreadCol, ssc.weightCol, ssc.tenorCol)
        s_i = pd.read_csv(dir_path + '/data/test_ngc_USD_20170217.csv')
        s_i['MatDate'] = s_i.MatDate.apply(lambda x: pd.to_datetime(x))
        s_i = u.process_data(s_i, ssc.curveCol, ssc.spreadCol, ssc.weightCol, ssc.tenorCol)
        res_f, res_d = ssc.fit_date_pair(ssc.end_date, s_i, s_prev)

        level = np.array((res_f['yout']))
        change = np.array((res_d['yout']))
        coupon = np.array(res_f['coupon'])
        AmtOut = np.array(res_f['AmtOutstanding_pdf'])
        sumAmtOut = res_f['SumAmtOutstanding']
        NumOutliersLvls = res_f['IssuerBonds'].loc[res_f['IssuerBonds'].outlier == True].shape[0]
        NumOutliersChgs = res_d['IssuerBonds'].loc[res_d['IssuerBonds'].outlier == True].shape[0]

        level_benchmark = [3.869896288,3.87396203,3.87842008,3.883413096,3.902457887,3.928742394,3.961868794,
                           3.99966851,4.039523728,4.107963378,4.139571893,4.132322506,4.093855461,4.033823569,
                           3.875157974,3.543684945,3.239545851,3.215541088,3.175099568,3.121768791,3.079908269,
                           3.045379252,2.990511224]

        change_benchmark = [0.012924828,0.012943643,0.012964622,0.012988766,0.013084007,0.013218963,0.013392714,
                            0.013604642,0.013844757,0.014323107,0.014721773,0.015019972,0.015276246,0.01552283,
                            0.015861111,0.015775485,0.015073519,0.014219347,0.012465087,0.011604379,0.011004523,
                            0.010508346,0.009720572]

        coupon_benchmark = [11.00281598,11.00281598,11.00281598,11.00281598,11.00281598,11.00281598,11.00281598,
                            11.00281598,11.00281598,10.96837654,10.93732609,10.91301569,10.89879642,10.89790249,
                            10.92211107,10.86439878,10.33332531,9.782111242,8.909383307,7.383239295,7.383239295,
                            7.383239295,7.383239295]

        AmtOut_benchmark = [0.111767954,0.590217265,2.388415436,7.757378803,98.42659118,442.2147921,1019.096472,
                            1550.361169,1968.045341,3803.143918,7758.7541,11753.52928,14694.97582,16773.24364,
                            16908.74676,6157.807287,381.7620225,40.80301767,1.065288563,0.001890128,3.6379E-06,
                            8.99744E-09,1.31925E-13]

        sumAmtOut_benchmark = 13890.871

        NumOutliersLvls_benchmark = 11
        NumOutliersChgs_benchmark = 1

        for i in range(0, len(ssc.tenorGrid)):
            self.assertAlmostEquals(level[i], level_benchmark[i], delta=1e-5)
            self.assertAlmostEquals(change[i], change_benchmark[i], delta=1e-5)
            self.assertAlmostEquals(coupon[i], coupon_benchmark[i], delta=1e-5)
            self.assertAlmostEquals(AmtOut[i], AmtOut_benchmark[i], delta=1e-2)

        self.assertAlmostEquals(sumAmtOut, sumAmtOut_benchmark, delta=1e-2)
        self.assertEquals(NumOutliersLvls, NumOutliersLvls_benchmark)
        self.assertEquals(NumOutliersChgs, NumOutliersChgs_benchmark)


    def test_xcompress(self):
        xout = np.arange(0.0,50.,1.)
        #xout = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100])
        xin = np.arange(5.0,25.0,1.0)

        xCompressLong=10.0
        xCompressShort=1.0
        xLast=max(xout)
        xFirst=min(xout)
        compressionLong=0.0
        compressionShort=0.0
        width=0.

        lxin = u.logx1Compress(xin, xCompressLong, xCompressShort, max(xout), min(xout), compressionLong, compressionShort,widthLong=0.,widthShort=0.)
        lxout = u.logx1Compress(xout, xCompressLong, xCompressShort, max(xout), min(xout), compressionLong, compressionShort, widthLong=0., widthShort=0.)

        expectedlxin = np.array([1.79175946923,
                                1.94591014906,
                                2.07944154168,
                                2.19722457734,
                                2.30258509299,
                                2.3978952728,
                                2.3978952728,
                                2.3978952728,
                                2.3978952728,
                                2.3978952728,
                                2.3978952728,
                                2.3978952728,
                                2.3978952728,
                                2.3978952728,
                                2.3978952728,
                                2.3978952728,
                                2.3978952728,
                                2.3978952728,
                                2.3978952728,
                                2.3978952728])

        expectedlxout = np.array([0.69314718056,0.69314718056,
                                1.09861228867,1.38629436112,
                                1.60943791243,1.79175946923,
                                1.94591014906,2.07944154168,
                                2.19722457734,2.30258509299,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728,
                                2.3978952728,2.3978952728])

        for i,x in enumerate(lxin):
            self.assertAlmostEquals(lxin[i], expectedlxin[i], delta=1e-6)

    def test_smoothfit_parabola(self):
        params = {'specific_curve': 'USD-JPM-SEN',
                  'start_date': '2017-02-16',
                  'end_date': '2017-02-17',
                  'Currency': 'USD',
                  'smfitCompressionShort': 0.,
                  'smfitCompressionLong': 0.,
                  'smfitSpldf':20,
                  'smfitLogxScale': False}

        ssc_i = ngc.SmoothSplineCurve(**params)
        x=np.arange(1.,20.,1.0)
        y=x**2.
        w=np.ones(len(x))

        df_fit = pd.DataFrame({'x':x,'y':y,'w':w})
        res = ssc_i.SmoothFit(df_fit)

        for i,e in enumerate(y):
            message = 'X**2 curve fails smoothfit, point %d'% x[i]
            self.assertAlmostEquals(y[i]/res['yout'][i],1. ,msg=message, delta=1e-6)

    def test_bond_transform_maturity(self):
        dfr = pd.read_csv(dir_path + '/data/xcTransform_test_rates.csv')
        df_test = pd.read_csv(dir_path + '/data/xcTransform_test.csv')

        results = u.calculate_bond_maturities(df_test.InYears, df_test.oas, df_test.coupon, dfr.InYears, dfr.r, freq=4.)

        input = np.array(df_test.maturity)
        output = np.array(results.maturity)

        for i in range(0,len(input)):
            message = 'Bond maturity calc failure'
            self.assertAlmostEquals(input[i], output[i],msg=message, delta=1e-8)

    def test_bond_transform_price(self):
        dfr = pd.read_csv(dir_path + '/data/xcTransform_test_rates.csv')
        df_test = pd.read_csv(dir_path + '/data/xcTransform_test.csv')

        results = u.calculate_bond_maturities(df_test.InYears, df_test.oas, df_test.coupon, dfr.InYears, dfr.r, freq=4.)
        results = u.calculate_bond_prices(results, dfr, freq=4.)

        for i,row in results.iterrows():
            message = 'Bond price calc failure'
            self.assertAlmostEquals(row['price'], 1.0,msg=message, delta=1e-8)

    def test_bond_transform_II(self):
        dfr = pd.read_csv(dir_path + '/data/xcTransform_test_rates.csv')
        df_test = pd.read_csv(dir_path + '/data/xcTransform_test.csv')
        freq = 4.

        results = u.transform_oas_to_spot(df_test, dfr, freq)

        price_test = []
        for i,row in results.iterrows():
            p = u.bond_price_oas_(row['maturity'], row['coupon'], 0., freq, results.maturity, results.total_spot)
            price_test.append(p)

        print price_test

        for p in price_test:
            message = 'Bond price calc failure'
            self.assertAlmostEquals(p, 1.0, msg=message, delta=1e-2)

    def test_needXccySupport_query_I(self):
        Currency = 'BRL'
        Region = 'GLOBAL'
        Industry = 'ALLSECTORS'
        needSupport = q.NeedXccySupport(Currency, Industry, Region)
        self.assertEquals(needSupport, True)

    def test_needXccySupport_query_II(self):
        Currency = 'USD'
        Region = 'GLOBAL'
        Industry = 'ALLSECTORS'
        needSupport = q.NeedXccySupport(Currency, Industry, Region)
        self.assertEquals(needSupport, False)

if __name__ == '__main__':
    unittest.main()