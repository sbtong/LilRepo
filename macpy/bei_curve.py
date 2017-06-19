import sys
import inspect
import os
import logging
from utils import Utilities
from CurveQueries import Curve_Database_Queries
from curve_utility import getAnalysisConfigSection, getOracleDatabaseInfo, getMSSQLDatabaseInfo
from utils.database import DatabaseExtract
from curve_utility import add_business_days
from dateutil import parser as dateparser
import pandas as pd
import datetime, operator
from dateflow import DateFlow
from xmltodict import parse

class BEI_Database_Queries(Curve_Database_Queries):
		def __init__(self, mktDBInfo, modelDBInfo, macDBInfo, use_node_quote_final = False, use_research_table = False):
				Curve_Database_Queries.__init__(self, mktDBInfo, modelDBInfo, macDBInfo, use_node_quote_final = use_node_quote_final, use_research_table = use_research_table)
				
				self.benchmark_curve_data_query = lambda bei_curve_id:"""select configuration from MarketData.dbo.CurveGenerationConfiguration where CurveId = %d"""%(bei_curve_id)
				
				self.curve_bond_prices_query    = lambda curve_id, start_date, end_date:"""SELECT B.InstrCode,TradeDate,Prc,DebtIsoCurCode, MatStdYld
																																 FROM QAI_Corrections..FIEJVCurveBenchmark_Corrected B
																																 JOIN qai.dbo.FIEJVPRCDly 
																																 on  B.InstrCode=FIEJVPRCDly.InstrCode
																																 AND TradeDate BETWEEN StartDate AND ISNULL(EndDate, '9999-12-31') 
																																 WHERE CurveId=%s
																																 AND TradeDate BETWEEN %s AND %s
																																 """%(curve_id, "'"+start_date+"'", "'"+end_date+"'")
				
				self.curve_bond_prices_query_by_country = lambda country_code, start_date, end_date:"""
						SELECT 
					  sec.InstrCode,
					  px.TradeDate,
					  px.Prc,
					  sec.DebtISOCurCode,
					  px.MatStdYld
					 from  
					  qai..FIEJVSecInfo sec
					  JOIN qai..FIEJVPrcDly px on px.InstrCode = sec.InstrCode and px.TradeDate >= '{start_date}' and px.TradeDate <= '{end_date}'
					  JOIN [QAI].[dbo].FIEJVOrgInfo org on org.GemOrgId=sec.IssrGEMOrgId
					  LEFT JOIN [QAI].[dbo].FIEJVSecIdent   id  on     id.InstrCode  = sec.InstrCode  
														and id.EndDate   is null 
														and id.Item       = 35  
														and id.SeqNum     = 1
					  INNER join [QAI].[dbo].FIEJVCpn cpiv on  cpiv.instrcode = sec.InstrCode and cpiv.Item = 120 -- fix rate
					  LEFT join [QAI].[dbo].FIEJVCpn cpnFreq  on cpnFreq.InstrCode = sec.InstrCode
																						  and cpnFreq.Item = 129 --CouponPaymentFrequencyCode
																						  and cpnFreq.CpnLegNum = 1
					  --LEFT  JOIN EJV_GovCorp.dbo.orig_iss_info ii on convert(varbinary(max), sec.EJVAssetId, 1) = ii.asset_id
					  JOIN [QAI].[dbo].FIEJVSecAmt  amt on    amt.Item       = 191 
					  and amt.InstrCode  = sec.InstrCode
								   and amt.AmtDate    = (select max(AmtDate) 
																	from [QAI].[dbo].FIEJVSecAmt amt2 
																	where amt2.InstrCode  = amt.InstrCode 
																		and amt2.Item       = 191  
																		and amt2.AmtDate   <= px.TradeDate)   
			   where 
					  sec.IssTypeCode = 'GOVT' and MatStdYld is not NULL and
					  sec.CurrCpnRate > 0.0 and
					  CAST(amt.value_ AS decimal) > 0.0 and
					  sec.SecInfoDFlag & 4096 != 0  -- Only Inflation Linked Bonds
					  and org.IsoCtryCode = '{country_code}'
			   ORDER BY DebtISOCurCode, TradeDate
	   """.format(country_code=country_code, start_date=start_date, end_date=end_date)


				self.coupon_info = lambda instrcode:"""SELECT IssName,MatDate,FIEJVSecInfo.CurrCpnRate,InitAccDate,LastCpnDate, FrstCpnDate,
														 DebtIssTypeCode 
																					from qai.dbo.FIEJVSecInfo JOIN qai.dbo.FIEJVCpnHdr on qai.dbo.FIEJVCpnHdr.InstrCode=qai.dbo.FIEJVSecInfo.InstrCode
																					WHERE FIEJVSecInfo.InstrCode=%d"""%(instrcode)
				
				self.bond_dateflow_info = lambda instrcode:r"""select CpnLegNum,Mnemonic,code.Desc_
																		from qai.dbo.FIEJVCpn as cpn
																		join qai.dbo.FIEJVItem as item on item.Item=cpn.Item
																		join qai.dbo.FIEJVCode as code on cpn.Value_=cast(code.Code as varchar(5))
																		join (
																				select FIEJVCode.Code,FIEJVItem.Item
																				from qai.dbo.FIEJVCode,qai.dbo.FIEJVItem
																				where
																				FIEJVCode.Type_=0
																				and
																				FIEJVItem.Desc_ like '% FIEJVCODE table (' + FIEJVCode.Desc_ + ')%'
																				and 
																				FIEJVItem.TableName='FIEJVCpn'
																		) as link on link.Item=cpn.Item and code.Type_=link.Code
																		where cpn.InstrCode={0}""".format(instrcode)
				
				self.calendar_info = lambda instrcode:r"""
															SELECT 
															DomSettDays
														, D.Desc_ as [Day Type]
														, H.Desc_ as [Holiday Calendar]
														, ExDivDays
														FROM qai.dbo.FIEJVSecCalendar
														JOIN qai.dbo.FIEJVDesc as D
															on qai.dbo.FIEJVSecCalendar.DomSettCalCode=D.Code
															AND D.Type_=(SELECT Code from qai.dbo.FIEJVDesc where Type_=0 AND Desc_='Day Type Code')
														LEFT JOIN qai.dbo.FIEJVDesc as H 
															on  qai.dbo.FIEJVSecCalendar.SettHldConvCode=H.Code
															AND H.Type_=(SELECT Code from qai.dbo.FIEJVDesc where Type_=0 AND Desc_='Holiday Calendar Code')
														where InstrCode={0}""".format(instrcode)
				
				self.bond_face_value_query = lambda instrcode:r"""SELECT Value_ from qai.dbo.FIEJVMiscFInfo
															WHERE InstrCode=%d
															and 
															Item=(select FIEJVItem.Item from qai.dbo.fiejvitem where Mnemonic='PriceQuoteBaseValue')"""%(instrcode)
				
				self.bond_price_real_or_nominal = lambda instrcode:"""DECLARE @QuoteType varchar(100)
																					exec [MarketData].[dbo].[BondQuotationType] @InstrCode = %d, @QuoteType = @QuoteType output
																					select  @QuoteType as QuoteType"""%(instrcode)
				
				self.index_factor_formula = lambda instrcode:"""SELECT IdxLkbckNum,IdxLkbckTypeCode,PayFmlIdxName,PayFmlNumVal,PayFmlOperator
																						FROM
																						(
																								SELECT
																									InstrCode,
																									Value_ as FormulaID,
																									RANK() OVER (PARTITION BY InstrCode ORDER BY EffDate DESC) as RNK
																								from qai.dbo.FIEJVPrincipalHdr
																								WHERE Item=(SELECT Item from qai.dbo.FIEJVItem where Mnemonic='PrinFmlID')
																						) as FORMULA
																						JOIN qai.dbo.FIEJVFormulaHdr on FmlID=FORMULA.FormulaID
																						WHERE FORMULA.RNK=1
																						and InstrCode=%d
																						ORDER BY PayFmlSeqNum"""%(instrcode)

def create_database_connections(environment):
	current_module = sys.modules[__name__]
	module_path = inspect.getabsfile(current_module)
	pwd = os.path.dirname(module_path)
	logConfigFile = pwd + "/log.config"
	if not os.path.exists(logConfigFile):
		raise Exception("Logging configuration file:%s does not exist."%logConfigFile)
	configFile = open(pwd+"/production.config",'r')
	configuration = Utilities.loadConfigFile(configFile)
	sectionID = getAnalysisConfigSection(environment)
	envInfoMap = Utilities.getConfigSectionAsMap(configuration, sectionID)
	mktDBInfo = getOracleDatabaseInfo(Utilities.getConfigSectionAsMap(configuration, envInfoMap.get('equitymarketdb', None)))
	modelDBInfo = getOracleDatabaseInfo(Utilities.getConfigSectionAsMap(configuration, envInfoMap.get('equitymodeldb', None)))
	macDBInfo = getMSSQLDatabaseInfo(Utilities.getConfigSectionAsMap(configuration, envInfoMap.get('macdb', None)))
	return mktDBInfo, modelDBInfo, macDBInfo

class BEI(DatabaseExtract):
		def __init__(self, country_code, start_date, end_date, environment, use_node_quote_final = False, use_research_table = False):
			mktDBInfo, modelDBInfo, macDBInfo = create_database_connections(environment)
			self.curve_queries = BEI_Database_Queries(mktDBInfo, modelDBInfo, macDBInfo, use_node_quote_final, use_research_table = use_research_table)
			self.curve_short_names = self.curve_queries.fetch_mac_curves_query_dataframe(self.curve_queries.curve_detail_by_curvetypes_country(['BEI.Zero'],country_code, start_date, end_date))['CurveShortName']
			self.country_code = country_code
			self.start_date = start_date
			self.end_date = end_date
		
		def calculate_bond_index_factor(self, bond_data):
				query = lambda ref_index:"""SELECT eff_dt,mkt_value
							 from EJV_rigs..rt_n_indx_level
							 WHERE indx_id=(SELECT indx_id FROM EJV_rigs..rt_n_indx where short_name='%s')"""%(ref_index)
				results = self.curve_queries.fetch_mac_curves_query_result(query(bond_data['PayFmlIdxName']))
				IdxLkbckTypeCode = bond_data['IdxLkbckTypeCode']
				IdxLkbckNum = bond_data['IdxLkbckNum']
				PayFmlNumVal = bond_data["PayFmlNumVal"]
				operators = {
						'+' : operator.add,
						'-' : operator.sub,
						'*' : operator.mul,
						'/' : operator.div,
						'%' : operator.mod,
						'^' : operator.xor
				}
				if IdxLkbckTypeCode == 'c':#calendar days
						_d = (pd.to_datetime(bond_data['settlement_date']) - datetime.timedelta(days=IdxLkbckNum)).date()
				elif IdxLkbckTypeCode == 'd':#business days
						_d = add_business_days(bond_data['settlement_date'], days=-IdxLkbckNum)
				elif IdxLkbckTypeCode == 'm':#months
						_d = (pd.to_datetime(bond_data['settlement_date']) - datetime.timedelta(days=30)).date()
				data = pd.DataFrame([{'effective_date':d[0], 'market_value':d[1] }for d in results])
				if len(data):
					data.sort('effective_date', inplace=True)
					_result = data[data['effective_date'] <= _d]['market_value'].tail(1)
					if len(_result):
							if (PayFmlNumVal is not None): # check for a possible case of reference index with no value
								index_value = _result.values[0]
								PayFmlOperator = bond_data["PayFmlOperator"]
								index_factor = operators[PayFmlOperator](index_value, PayFmlNumVal) if not bond_data['quote_type'] == 'real' else 1.0
							else:
								index_factor = 1.0
							bond_data.update({'index_factor':index_factor})
							bond_data.update({'notional':bond_data['face_value'] * index_factor})
					else:
						raise Exception("No data found for effective date below settlement date")
				else:
					logging.warning('No inflation linked index was found %s'%(self.country_code))
					index_factor = 1.0
					bond_data.update({'index_factor':index_factor})
					bond_data.update({'notional':bond_data['face_value'] * index_factor})
				return bond_data
		
		def get_coupon_frequency(self, description):
				if description == 'Semiannually':
						return 2
				elif description == 'Annually':
						return 1
				return 2
		
		def get_bond_cash_flow_info(self, bond_data):
				for cpn_leg_num, mnemonic, description in self.curve_queries.fetch_mac_curves_query_result(self.curve_queries.bond_dateflow_info(bond_data['InstrCode'])):
						bond_data.setdefault('coupon_leg_num', cpn_leg_num)
						if mnemonic == 'DayCountCode':
								bond_data.setdefault('day_count_method', description)
						elif mnemonic == 'CouponPaymentFrequencyCode':
								bond_data.setdefault('description', description)
				return bond_data
		
		def extract_indexbase_results(self, results):
				data = pd.DataFrame([{'IdxLkbckNum':d[0],
								 'IdxLkbckTypeCode':d[1],
								 'PayFmlIdxName':d[2],
								 'PayFmlNumVal':d[3],
								 'PayFmlOperator':d[4]}for d in results])
				IdxLkbckNum = data.dropna(subset=['IdxLkbckNum'])['IdxLkbckNum'].values[0]
				IdxLkbckTypeCode = data.dropna(subset=['IdxLkbckTypeCode'])['IdxLkbckTypeCode'].values[0]
				PayFmlIdxName = data.dropna(subset=['PayFmlIdxName'])['PayFmlIdxName'].values[0]
				PayFmlNumVal = data.dropna(subset=['PayFmlNumVal'])['PayFmlNumVal'].values[0]
				PayFmlOperator = data.dropna(subset=['PayFmlOperator'])['PayFmlOperator'].values[0]
				return IdxLkbckNum, IdxLkbckTypeCode, PayFmlIdxName, PayFmlNumVal, PayFmlOperator
		
		def extract_from_db(self):

			def filter_bad_bonds(curve_id, curve_quotes):
				if len(curve_quotes):
					bad_bonds_data = pd.DataFrame([{'TradeDate':dateparser.parse(trade_date).date(), 'InstrCode':instrcode, 'ItemId':itemid} for trade_date, instrcode, itemid in
																																self.curve_queries.fetch_mac_curves_query_result("""select TradeDate, InstrCode, ItemId
																																from MarketData.dbo.DerivCurveFilteredBondCorrection where CurveId=%d"""%(curve_id))])
					if len(bad_bonds_data):
						recurring_bad_bonds = bad_bonds_data[bad_bonds_data['ItemId'] == 14]['InstrCode'].values
						if len(recurring_bad_bonds):
							curve_quotes = curve_quotes[curve_quotes['InstrCode'].map(lambda x:x not in recurring_bad_bonds)]
						dated_bad_bonds = bad_bonds_data[bad_bonds_data['ItemId'] == 1]
						if len(dated_bad_bonds):
							merged_df = curve_quotes.merge(dated_bad_bonds, how='left', on = ['InstrCode', 'TradeDate'])
							filtered_df = merged_df[pd.isnull(merged_df['ItemId'])]
							return filtered_df
				return curve_quotes

			results = []
			for curve_short_name in self.curve_short_names:
				bei_curve_data = self.curve_queries.fetch_mac_curves_query_result(self.curve_queries.curve_detail_by_shortname(curve_short_name))
				bei_curve_id = bei_curve_data[0][5]
				bei_curve_currency = bei_curve_data[0][1]
				curve_nodes_data = [{'tenorenum':data[0],'activetodate':data[1],'activefromdate':data[2],'curvenodeid':data[3],'curveid':data[4] 
												}for data in self.curve_queries.fetch_mac_curves_query_result("""select TenorEnum, ActiveToDate, ActiveFromDate, 
																																		 CurveNodeId, CurveId
																																		 from MarketData.dbo.CurveNodes 
																																		 where curveid = %d"""%(bei_curve_id))]
				benchmark_data = self.curve_queries.fetch_mac_curves_query_result(self.curve_queries.benchmark_curve_data_query(bei_curve_id))
				if len(benchmark_data):
					_benchmark_curveid = "<?xml version='1.0'?><head>"+benchmark_data[0][0]+"</head>"
					benchmark_curveid = int(parse(_benchmark_curveid).get('head',{}).get('EJVBenchmarkCurveId'))
					curve_bonds_prices_data = [ {'InstrCode':data[0], 'TradeDate':data[1].date(), 'Price':data[2], 'DebtISOCurCode':data[3], 'MatStdYld':data[4]}
																	for data in self.curve_queries.fetch_mac_curves_query_result(self.curve_queries.curve_bond_prices_query(benchmark_curveid,self.start_date, self.end_date))]
				else:
					curve_bonds_prices_data = [{'InstrCode':data[0], 'TradeDate':data[1].date(), 'Price':data[2], 'DebtISOCurCode':data[3], 'MatStdYld':data[4]}
																	for data in self.curve_queries.fetch_mac_curves_query_result(self.curve_queries.curve_bond_prices_query_by_country(self.country_code, self.start_date, self.end_date))]
				for data in curve_bonds_prices_data:
						issname, matdate, firstcpnrate, initaccdate, lastcpndate, frstcpndate, DebtIssTypeCode = self.curve_queries.fetch_mac_curves_query_result(self.curve_queries.coupon_info(data['InstrCode']))[-1]
						data = self.get_bond_cash_flow_info(data)
						dom_settlement_days, day_type, holiday_calendar,ex_div_days = self.curve_queries.fetch_mac_curves_query_result(self.curve_queries.calendar_info(data['InstrCode']))[-1]
						_face_value = self.curve_queries.fetch_mac_curves_query_result(self.curve_queries.bond_face_value_query(data['InstrCode']))
						face_value = _face_value[0][0] if len(_face_value) else 100.0
						quoteType_result = self.curve_queries.fetch_mac_curves_query_result(self.curve_queries.bond_price_real_or_nominal(data['InstrCode']))
						try:
							IdxLkbckNum,IdxLkbckTypeCode,PayFmlIdxName,PayFmlNumVal,PayFmlOperator = self.extract_indexbase_results(self.curve_queries.fetch_mac_curves_query_result(self.curve_queries.index_factor_formula(data['InstrCode'])))
						except KeyError:
							IdxLkbckNum,IdxLkbckTypeCode,PayFmlIdxName,PayFmlNumVal,PayFmlOperator = 0, 'c', None, None, None
						data.update({'IssName':issname, 'MatDate':matdate.date(), 'InitAccDate':initaccdate.date(),
												 'DebtIssTypeCode':DebtIssTypeCode,
												 'LastCpnDate':lastcpndate.date(),
												 'FrstCpnDate': frstcpndate.date(),
												 'CompFreqCode':self.get_coupon_frequency(data['description']),
												 'DomSettDays':dom_settlement_days,
												 'settlement_date':add_business_days(data['TradeDate'], days=dom_settlement_days),
												 'day_type':day_type, 'holiday_calendar':holiday_calendar,
												 'ex_div_days':ex_div_days, 'FrstCpnRate':firstcpnrate,
												 'quote_type' :quoteType_result[0][0] if len(quoteType_result) else None,
												 'face_value':face_value,
												 'IdxLkbckNum' : IdxLkbckNum,
												 'IdxLkbckTypeCode' : IdxLkbckTypeCode,
												 'PayFmlIdxName' : PayFmlIdxName,
												 'PayFmlNumVal' : PayFmlNumVal,
												 'PayFmlOperator' : PayFmlOperator,
												 'TimeInYears' : (matdate.date() - initaccdate.date()).days/365.25,
												 'CurveShortName' : curve_short_name,
												 'NomTermToMat' : (matdate.date() - data['TradeDate']).days/365.25,
												 'PriceInclAccrIntFlg':'n'
												})
						
						self.calculate_bond_index_factor(data)
				curve_bonds_prices_data = pd.DataFrame(curve_bonds_prices_data)
				if len(curve_bonds_prices_data):
					curve_bonds_prices_data.sort('NomTermToMat', inplace=True)
					curve_bonds_prices_data = filter_bad_bonds(bei_curve_id, curve_bonds_prices_data)
				results.append(curve_bonds_prices_data)
			return results

class BEI_DateFlow(DateFlow):
	def __init__(self, coupon_rate, issue, maturity, valuation, index_factor = 1.0, freq=2, first_cpn_dt=None, last_cpn_dt=None, legacy=False):
		super(DateFlow, self).__init__(coupon_rate, issue, maturity, valuation,freq, first_cpn_dt, last_cpn_dt, legacy)
		self.index_factor = index_factor

	def get_cash_flows(self):
		self.coupon_rate = self.coupon_rate * self.index_factor * 100.0
		n = len(self.future_dates)
		self.cash_flows = [0.0]
		for i in range(n)[1:]:
			if i == n - 1:
				self.cash_flows.append(max(100.0 * self.index_factor, 100.0) + self.coupon_rate) 
			else:
				self.cash_flows.append(self.coupon_rate)


def test_input_bonds(curve_short_name, start_date, end_date, environment='DEV'):
	bei_curve_instance = BEI(curve_short_name = curve_short_name, start_date = start_date, end_date = end_date, environment = environment)
	bei_curve_data = bei_curve_instance.curve_queries.fetch_mac_curves_query_result(bei_curve_instance.curve_queries.curve_detail_by_shortname(curve_short_name))
	bei_curve_id = bei_curve_data[0][5]
	query = "select  * from DerivedDataInflationIntermediates2 where CurveId = %d and TradeDate >= '%s' and TradeDate <= '%s'"%(bei_curve_id, start_date, end_date)
	actual_df = bei_curve_instance.curve_queries.fetch_mac_curves_query_dataframe(query)
	expected_df = bei_curve_instance.extract_from_db()
	assert len(actual_df) == len(expected_df)
	assert len(set(actual_df['InstrCode'].values).difference(expected_df['InstrCode'].values)) == 0