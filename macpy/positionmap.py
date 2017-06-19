from __future__ import print_function
import time
import ConfigParser
import optparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup    
from suds.client import Client
from StringIO import StringIO



class PositionMapperWebService(object):
    def __init__(self):
        self.serviceurl = 'http://tethys/ClientIntercom_DEV_MAC_UATDB/GetDataFromDbService.svc?singleWsdl'
        self.db = 'MarketData'
        self.storedProc = 'IC_DataRetrieval_SingleSecurity'

    def request_xml_item(self, instrumentId, **kwargs):
        instrumentList = instrumentId if isinstance(instrumentId, list) else [instrumentId]
        result = [x for x in self.request_xml_iterable(instrumentId, **kwargs)]
        if(len(result)==1): return result[0]
        return result


    def request_xml_iterable(self, instrumentId, secIdType='ISIN', analysisDate='05/02/15', dbserver='prod_mac_mkt_db'):
        """
        secIdType = ISIN|MODELDB_ID|CUSIP|TICKER|SEDOL

        """

        self.dbserver = dbserver
        self.analysisDate = analysisDate
        client =  Client(self.serviceurl, timeout=3000)
        extractJob = client.factory.create('GetDataFromDb') 
        self.secIdType = secIdType
        extractJob.request.SecIdType=self.secIdType
        extractJob.request.AsOfDate=datetime.strptime(analysisDate, '%m/%d/%y')
        extractJob.request.DatabaseServerName=self.dbserver
        extractJob.request.DatabaseName=self.db
        extractJob.request.StoredProc=self.storedProc
        extractJob.request.IncludeSqlInOutput=False
        extractJob.request.OutputCsvFormat=False

        instrumentList = instrumentId if isinstance(instrumentId, np.ndarray) or isinstance(instrumentId, list)  else [instrumentId]

        for x in instrumentList:
            secId = x
            extractJob.request.SecId=secId
            resp = client.service.GetDataFromDb(extractJob.request)
            extractedXML = resp.RawDataStringFromDatabase
            yield BeautifulSoup(extractedXML)


                                                                         


testSample = """Positions	ClientDescription
US61690FAQ63	MSBAM 2015-C22 B
US90269GAS03	UBSCM 2012-C1 F
US92936TAM45	WFRBS 2012-C7 G
US12626BAY02	COMM 2013-CR10 E
US36197XAC83	GSMS 2013-GC12 E
US78413MAQ15	SFAVE 2015-5AVE D
US43289UAL98	HILT 13-HLF EFL
US26885KAQ31	EQTY 2014-INNS F
US61974QAS57	MOTEL 2015-MTL6 E
US21870KAQ13	CORE 2015-WEST F
US46642KAC62	JPMCC 2014-FRR1 BK10
US12592PBG72	COMM 2014-UBS6 XA
US38378KRX99	GNR 2013-78 IO
US37734UAA60	GLCT 2015-B A
XS0280699793	ALTIS 2006-1X A
ES0338186010	UCI 16 A2
XS1192475785	ALBA 2015-1 E
XS0304286254	NGATE 2007-2X DB
US3137AR2C92	FHR 4059 SA
US3136A9HW18	FNR 2012-110 LS
US3136AAZP33	FNR 2012-133 JS
US3136AARZ07	FNR 2012-134 SD
US3136AJQA73	FNR 2014-19 MS
US3136AMNR64	FNR 2015-3 TS
US31398WSK17	FHR 3632 SA
US31398SRX35	FNR 2010-143 SB
US3137AX3W12	FHR 4142 BI
US38377YJB74	GNR 2011-132 GI
US38378EW903	GNR 2012-75 AI
US3137B3JJ80	FHR 4231 S
US3136A7RN43	FNR 2012-90 SA
US3136ALCJ84	FNR 2014-63 SL
US31396RAV96	FHR 3156 PS
US31397GES57	FHR 3309 SG
US31397YYU99	FHR 3510 AS
US3137AWX420	FHR 4138 S
US31397UTY54	FNR 2011-63 ST
US3136A0UE53	FNR 2011-78 ST
US3136AHG618	FNR 2013-130 SN
US3136AJQR09	FNR 2014-14 AS
US3136AJTT38	FNR 2014-20 SB
US31398VWB88	FHR 3653 AS
US3137AKVG35	FHR 3981 SQ
US3136AF6C38	FNR 2013-90 SD
US38374LK837	GNR 20052-65 SI
US38374MZK88	GNR 2006-10 SL
US38376FJN33	GNR 2009-61 WQ
US38376CZR32	GNR 2009-76 XS
US38378EWC38	GNR 2012-60 SG
US38379AG951	GNR 2014-43 IN
US38379BMG04	GNR 2014-60 IB
US38378FBW95	GNR 2013-10 DI
US38379CMN38	GNR 2014-84 AI
US3137B1D661	FHR 4192 US
US3137B5X652	FHR 4267 PO
US3136AEGC52	FNR 2013-53 JS
US3136AE5N39	FNR 2013-74 ES
US94986ABM99	WFMBS 2007-8 2A12
US881561L512	TMTS 2005-18AL PX
US81744XAA63	SEMT 2012-5 A
US12668BKA07	CWALT 06-J1 1A4
US251510HQ96	DBALT 2005-5 1A5
US32051HAD52	FHAMS 2006-FA3 A4
US761118MJ43	RALI 2005-QS16 A9
US93934FBR29	WMALT 2005-7 2CB6
US93363QAA67	WAMU 2006-AR15 1A
US16162WPB99	CHASE 2005-A1 1A1
US94982WAK09	WFMBS 2005-9 1A10
US05948KM892	BOAA 2005-9 1CB2
US17309KAJ60	CMALT 2006-A3 1A9
US251511AK72	DBALT 2006-AB2 A8
US362375AE70	GSAA 2006-10 AF5
US74162FAB13	PRIME 2007-1 A2
US75115EAA10	RALI 2006-QS11 1A1
US76114QAB14	RAST 2007-A7 A2
US86359DJZ15	SASC 2005-14 3A1
US040104RV54	ARSI 2006-W2 A2B Mtge
USD18AFVAA72	CRFT 2012-4X FUND
XS0795158251	ORYZA 1 A 12/28/2016
XS0612404623	SEALN 2011-1X A
XS1181280790	BALLSBRIDGE REPACKAGING 0 12/31/2049
US00432CAW10	ACCSS 2003-A B
US63544KAA43	NCFCT 2007-4 A3L
US63544LAA26	NCFCT 2007-4 A3R2
US63543XAD12	NCSLT 2007-1 A4
US80705V2007	SCHOL 2012-A R
US00164KAA51	ALM 2014-14A A1
US67108WAA71	OZLM 2014-7A A1A
XS1209504106	GROSV 2015-1X C
US38137EAE59	GOLD7 2013-7A SUB
USG6866JAB73	OZLMF 2013-3X SUB
USG6863CAA74	OZLM 2014-8X D
USG20674AC43	CECLO 2014-16XR DR
XS0312467680	GROSV_III-X D
XS0143894151	CLARE ISLAND BV 03/19/2020
XS0247039679	WODST II-X M2
"""


testhtml = """<html><body><toplevelelement rawdatatype="RawDataDictionary" rawdataversion="1" sourcedbserver="dev_mac_db_ny" 
<asofdate>05/02/2015 00:00:00</asofdate>
<axiomadataid>202395054</axiomadataid>
<categorynameenum>Structured Debt</categorynameenum>
<datacontentenum>Holding</datacontentenum>
<dataid>RefType|SecId|SecIdType=Holding|US61690FAQ63|ISIN</dataid>
<errormsg></errormsg>
<secid>US61690FAQ63</secid>
<secidtype>ISIN</secidtype>
<termsandconditions rawdatatype="RawDataDictionary">
<axiomadataid>202395054</axiomadataid>
<call></call>
<categorynameenum>Structured Debt:CMBS</categorynameenum>
<couponschedule rawdatatype="RawDataList">
<couponschedule_element rawdatatype="RawDataDictionary">
<axiomadataid>202395054</axiomadataid>
<couponfromdate>04/15/2015 00:00:00</couponfromdate>
<couponpaymentbusinessdayconvention>Following</couponpaymentbusinessdayconvention>
<couponpaymentcalendar>BAD MAPPING - B4D</couponpaymentcalendar>
<couponpaymentcalendarnumber>11</couponpaymentcalendarnumber>
<couponpaymentcurrency>USD</couponpaymentcurrency>
<couponpaymentfrequency>Monthly</couponpaymentfrequency>
<coupontype>Fixed</coupontype>
<daycountconvention>D30Y360US</daycountconvention>
<errormsg></errormsg>
<fixed rawdatatype="RawDataDictionary">
<couponrateinpct>3.883</couponrateinpct>
</fixed>
<float></float>
<iscomplexcoupon>0</iscomplexcoupon>
<resetrates></resetrates>
<trancheid>0x001005AD990F02CF</trancheid>
</couponschedule_element>
</couponschedule>
<currency>USD</currency>
<cusip>61690FAQ6</cusip>
<dateddate>04/01/2015 00:00:00</dateddate>
<firstcoupondate>05/15/2015 00:00:00</firstcoupondate>
<firstcouponstartaccruingdate>04/01/2015 00:00:00</firstcouponstartaccruingdate>
<holdingprice rawdatatype="RawDataDictionary">
<currency>USD</currency>
<date>05/01/2015 00:00:00</date>
<isrolledover>Y</isrolledover>
<price>101.93</price>
<type>Clean</type>
</holdingprice>
<holidays>New York</holidays>
<iscallable>Y</iscallable>
<isconvertible>N</isconvertible>
<isin>US61690FAQ63</isin>
<isindexlinked>N</isindexlinked>
<isputtable>N</isputtable>
<issinkable>N</issinkable>
<issue>MSBAM</issue>
<issuedate>04/10/2015 00:00:00</issuedate>
<issuername>MORGAN STANLEY BANK OF AMERICA MERRILL LYNCH TRUST</issuername>
<isunverifiedorpartialdata>N</isunverifiedorpartialdata>
<maturitydate>04/17/2048 00:00:00</maturitydate>
<modelingassumptions rawdatatype="RawDataDictionary">
<bondspreadcurveratingdataid>RefType|Name=BondSpreadCurve|USD.(SUB-IG).SPRSWP</bondspreadcurveratingdataid>
<bondspreadcurvesovzerodataid>RefType|Name=YieldCurve|US.USD.GVT.ZC</bondspreadcurvesovzerodataid>
<bondspreadcurveswapzerodataid>RefType|Name=BondSpreadCurve|US.USD.SWP.ZCS</bondspreadcurveswapzerodataid>
<moodysadvancedmodel></moodysadvancedmodel>
</modelingassumptions>
<name>MORGAN STANLEY BANK OF AMERICA MERRILL LYNCH TRUST</name>
<notional>100</notional>
<put></put>
<reportingattributes rawdatatype="RawDataDictionary">
<axiomadataid>202395054</axiomadataid>
<categorynameenum>Structured Debt:CMBS</categorynameenum>
<country>US</country>
<currency>USD</currency>
<gicslevel1>Structured Debt</gicslevel1>
<gicslevel2>Structured Debt:CMBS</gicslevel2>
<issuername>MORGAN STANLEY BANK OF AMERICA MERRILL LYNCH TRUST</issuername>
<issuertype>Structured Debt</issuertype>
<maturity>04/17/2048 00:00:00</maturity>
<name>MSBAM</name>
<rating>NR</rating>
</reportingattributes>
<seniority>Subordinated</seniority>
<series>15C22</series>
<sink></sink>
<tranche>B</tranche>
<trancheid>0x001005AD990F02CF</trancheid>
</termsandconditions>
<underlyingproc1></underlyingproc1>
<vendordataidtypeenum>TrancheAssetId</vendordataidtypeenum>
<vendordatainternalid>0x001005AD990F02CF</vendordatainternalid>
<xref rawdatatype="RawDataArray">
<columntitles>AxiomaDataId,SecIdType,SecId,FromDate,ToDate</columntitles>
<columntypes>int?,string,string,DateTime?,DateTime?</columntypes>
<row0>202395054,Cusip,61690FAQ6,04/23/2015 03:46:04,12/31/9999 00:00:00</row0>
<row1>202395054,ISIN,US61690FAQ63,04/23/2015 03:47:13,12/31/9999 00:00:00</row1>
</xref>
</toplevelelement></body></html>"""

def fake_iterator():

    i = 0
    max = 10
    while i < max:
        yield BeautifulSoup(testhtml)
        time.sleep(.1)
        i = i + 1




def main():
    import sys
    dbserver = 'devtest_mac_db_ny'
    secIdType='ISIN'
    analysisDate='05/02/15'

    instrumentDataFrame = pd.DataFrame.from_csv(StringIO(testSample), sep='\t')

    webservice = PositionMapperWebService()

    #for xmlResult in webservice.request_xml_iterable(instrumentDataFrame.index, secIdType, analysisDate, dbserver):
    for xmlResult in fake_iterator():
        secid = xmlResult.secid.text
        position = instrumentDataFrame.ix[secid]
        if(xmlResult.reportingattributes is None):
            print(position.name + '||' + position.values[0] + '|Not found in MarketData', sys.stdout )
        else:
            print(position.name + '|' + xmlResult.reportingattributes.axiomadataid.text + '|' + xmlResult.reportingattributes.categorynameenum.text + '|' + xmlResult.reportingattributes.issuername.text)


if __name__ == '__main__':
    main()

      
