import os, datetime, multiprocessing, sys
from collections import Counter
import argparse
import logging
from functools import partial
from dateutil import parser as date_parser
from itertools import groupby
import traceback
from scipy import interp
import numpy as np
import pandas as pd

from utils import Utilities
from CurveQueries import Curve_Database_Queries
from CurveQueries import extract_dataframe
from curve_utility import getAnalysisConfigSection, getOracleDatabaseInfo, getMSSQLDatabaseInfo

from collections import namedtuple

EconomyDateRange = namedtuple('EconomyDateRange', 'country_code currency_code start_date end_date')
EconomyDateRange.__new__.__defaults__ = (None,) * len(EconomyDateRange._fields) # default args to None
DebtSpec = namedtuple('DebtSpec', 'iss_type_code org_type_code issuer_id snrty_code top_composite_grade_incl bottom_composite_grade_incl')
DebtSpec.__new__.__defaults__ = (None,) * len(DebtSpec._fields)
CurveSpec = namedtuple('CurveSpec', 'composite_rating sector issuer_id')
CurveSpec.__new__.__defaults__ = (None,) * len(CurveSpec._fields)



class CurveGenerator(object):

    bonds_cache = None
    curves_cache = {}
    curves_metadata = {}
    curves_input_contracts = {}
    zero_curves_cache = {}
    corp_dataframe_cache = {}
    
    curve_family_child_parent_tree = {
                                        'RtgSprSv' : 'Sov.Zero',
                                        'GiSprRat' : 'RtgSprSv',
                                        'IssrSpr' :  'GiSprRat',
                                        'ProvSprSv' : 'Sov.Zero',
                                        'CvRatSpr' : 'Sov.Zero',
                                        'SupnRatSpr' : 'Sov.Zero',
                                        'AgncySprSv' : 'Sov.Zero',
                                        'SvSpr' : 'Sov.Zero',
                                        'SwapZrSpr' : 'Sov.Zero'
                                     }
    exit_status = 0

    @classmethod
    def create_curve_generator(cls, environment='DEV'):
        """
        Returns an instance of the curve generator engine with an underlying connection to the database environment
        """

        # Configuration boilerplate
        pwd = os.path.dirname(os.path.realpath(__file__)) + os.sep
        config_file = open(pwd+"production.config",'r')
        configuration = Utilities.loadConfigFile(config_file)
        section_id = getAnalysisConfigSection(environment)
        env_info_map = Utilities.getConfigSectionAsMap(configuration, section_id)
        market_db_info = getOracleDatabaseInfo(
            Utilities.getConfigSectionAsMap(configuration, env_info_map.get('equitymarketdb', None)))
        model_db_info = getOracleDatabaseInfo(
            Utilities.getConfigSectionAsMap(configuration, env_info_map.get('equitymodeldb', None)))
        mac_db_info = getMSSQLDatabaseInfo(
            Utilities.getConfigSectionAsMap(configuration, env_info_map.get('macdb', None)))

        generator = CurveGenerator(market_db_info, model_db_info, mac_db_info)

        return generator

    def __init__(self, mktDBInfo=None, modelDBInfo=None, macDBInfo=None, use_research=True, test_only=False,
                 delete_existing_quotes=True, calculate_composite_rating=False, run_cmf=False, run_SvSprEmg = False, country_code = None, currency_code = None, curve_types = None):
        self.curve_queries = Curve_Database_Queries(mktDBInfo, modelDBInfo, macDBInfo, use_research_table=use_research)
        self.delete_existing_quotes = delete_existing_quotes
        self.use_research_table = use_research
        self.calculate_composite_rating = calculate_composite_rating
        self.curve_node_quote_table = "CurveNodeQuote_Research" if self.use_research_table else "CurveNodeQuote"
        self.derived_curve_filtered_bond_table = "DerivCurveFilteredBond_Research" if self.use_research_table else "DerivCurveFilteredBond"
        self.derived_curve_filtered_bond_table_abs = "DerivCurveABS_Research" if self.use_research_table else "DerivCurveABS"
        self.derived_curve_filtered_bond_table_cmf = "DerivCMFCurveFilteredBond_Research" if self.use_research_table else "DerivCMFCurveFilteredBond" 
        self.test_only = test_only
        self.run_cmf = run_cmf
        self.run_SvSprEmg = run_SvSprEmg
        self.run_curves_by_currency = True if currency_code is not None else False
        self.run_curves_by_country = True if country_code is not None else False
        self.run_curves_by_type = True if curve_types is not None else False

    def calender_days_between(self, start_date, end_date):
        """
        :param start_date:datetime 
        :param end_date: datetime
        :return: datetime
        """
        current_date = start_date
        while current_date <= end_date:
            yield current_date
            current_date += datetime.timedelta(days=1)

    def run_curve(self, curve_short_name, trade_start_date, trade_end_date):
        # get curve info (countryISO and curveType)
        logging.info('Running %s' % curve_short_name)
        curve_data = {}

        self.curve_queries.regenerate_currency_rating_curve = True

        query = self.curve_queries.curve_detail_by_shortname(curve_short_name)
        results_list = self.curve_queries.fetch_mac_curves_query_result(query)
        if len(results_list) == 0:
            logging.warning('Curve Short Name %s does not exist.' % curve_short_name)
            return curve_data

        curve_metadata_record = self.curve_queries.fetch_mac_curves_query_result(query)
        curve_metadata = curve_metadata_record[0]
        currency_code = curve_metadata[1]
        country_code = curve_metadata[0]
        country_iso = country_code if country_code is not None else currency_code[0:2]
        curve_family = curve_metadata[2]
        composite_rating = curve_metadata[3]
        curve_id = curve_metadata[5]
        sector_enum = curve_metadata[6]
        underlying_curve_id = curve_metadata[7]
        issuer_id = curve_metadata[8]
        futures_contract_code = curve_metadata[9]
        underlying_type_enum = curve_metadata[10]
        curve_long_name = curve_metadata[11]
        ric = curve_metadata[12]
        active_to_date = date_parser.parse(curve_metadata[13])
        active_from_date = date_parser.parse(curve_metadata[14])

        econ_date = EconomyDateRange(country_code=currency_code[0:2],
                                     currency_code=currency_code,
                                     start_date=trade_start_date,
                                     end_date=trade_end_date)

        curve_spec = CurveSpec(composite_rating=composite_rating, sector=sector_enum, issuer_id=issuer_id)


        isABS = False
        isCMF = False
        isCDX = False

        market_status_enum = self.curve_queries.fetch_mac_curves_query_result(
            r"""select cv.MarketStatusEnum
                from MarketData.dbo.Curve cv
                where cv.CurveShortName='%s'""" % curve_short_name)[0][0]

        if underlying_curve_id is not None:
            curve_detail_query = self.curve_queries.curve_detail_by_curve_id(underlying_curve_id)
            underlying_curve_short_name_record = self.curve_queries.fetch_mac_curves_query_result(curve_detail_query)
            underlying_curve_short_name = underlying_curve_short_name_record[0][4]
        else:
            underlying_curve_short_name = None

        # check for curves that signal content central failure
        active_curves = extract_dataframe(self.curve_queries, "exec [DerivedDataActiveCurves] @AsOfDate='{}'".format(trade_start_date))

        isactive = curve_short_name in active_curves.CurveShortName.values

        CurveGenerator.curves_metadata.setdefault(curve_short_name,{}).update({'curve_id':curve_id,
                                                                               'parent_curve_short_name':underlying_curve_short_name,
                                                                               'sector':sector_enum,
                                                                               'isactive':isactive})

        if curve_family == 'Futures.CM':
            if self.run_cmf:
                isCMF = True
                curve_data = self.get_constant_maturity_futures_quotes(curve_short_name, futures_contract_code, curve_id, currency_code, trade_start_date, trade_end_date)
            else:
                logging.info("Curve %s was not generated"%curve_short_name)
                return curve_data
        elif curve_family in 'CDX.BasCor':
            isCDX = True
            curve_data = self.getBaseCorrelations(curve_short_name, curve_id, curve_long_name, trade_start_date, trade_end_date)
        elif curve_family in 'SwapZrSpr':
            #curve_data = self.get_swap_spread_quotes(curve_short_name, country_iso, curve_id, trade_start_date, trade_end_date)
            curve_data =  self.calculate_swap_spreads(curve_short_name, country_iso, currency_code, curve_id, trade_start_date, trade_end_date)
        elif curve_family in 'SwapZrSpr.v2':
            curve_data = self.get_swap_spread_quotes_v2(curve_short_name, country_iso, currency_code, curve_id, trade_start_date, trade_end_date, ric)
        elif curve_family in 'SwapZC':
            curve_data = self.get_irswap_yield_quotes(curve_short_name, country_iso, currency_code, curve_id, trade_start_date, trade_end_date, ric)
        elif curve_family in 'SwapZC.OIS':
            curve_date = self.get_oiswap_yield_quotes(curve_short_name, country_iso, currency_code, curve_id, trade_start_date, trade_end_date, ric)
        elif curve_family in 'SwapZCFwd':
            curve_data = self.get_fdswap_yield_quotes(curve_short_name, country_iso, currency_code, curve_id, trade_start_date, trade_end_date, underlying_type_enum[-2:])
        elif curve_family == 'RtgSprSv':
            curve_data = self.get_corp_rating_spread_quote(curve_short_name, econ_date, curve_spec)
        elif curve_family == 'GiSprRat':
            curve_data = self.get_corp_sector_spread_quote(curve_short_name, econ_date, curve_spec)
        elif curve_family == 'IssrSpr':
            curve_data = self.get_corp_issuer_spread_quote(curve_short_name, econ_date, curve_spec)
        elif curve_family == 'CvRatSpr':
            curve_data = self.get_covered_bond_quotes(curve_short_name, country_iso, currency_code, composite_rating, market_status_enum, trade_start_date, trade_end_date, curve_id)
        elif curve_family == 'ProvSprSv':
            curve_data = self.get_CAD_provincial_curve_quotes(curve_short_name, econ_date, curve_spec)
        elif curve_family == 'SupnRatSpr':
            curve_data = self.get_supranational_curve_quotes(curve_short_name, econ_date, curve_spec)
        elif curve_family == 'AgncySprSv':
            curve_data = self.get_agency_curve_quotes(curve_short_name, econ_date, curve_spec)
        elif curve_family in ('AFLP','AIRL','ALEA','AUTO','BIKE','BOAT','CARD','CBO','CLO','CMBS','CMO','COO','EQIP','EXIM','HLOC',
                              'HOME','MANU','NIM','OTHR','RECR','RMBS','STUD'):
            valid_abs = ['AIRL', 'AUTO', 'CARD', 'CMBS', 'CMO', 'EQIP', 'RMBS', 'HOME', 'MANU' ]
            isABS = True
            curve_data = self.get_abs_curve_quotes(curve_short_name, econ_date)
        elif curve_family == 'SvSpr':
            curve_data = self.get_sovereign_spread_quotes(curve_short_name, currency_code, country_iso, trade_start_date, trade_end_date)
        elif curve_family == 'SvSprEmg':
            if self.run_SvSprEmg:
                curve_data = self.get_sovereign_spread_quotes_emg(curve_short_name, currency_code, country_iso, trade_start_date, trade_end_date)
            else:
                logging.info("Curve %s was not generated"%curve_short_name)
                return curve_data
        else:
            logging.info("Curve %s was not generated"%curve_short_name)
            return curve_data
        CurveGenerator.curves_metadata.setdefault(curve_short_name,{}).update({'isABS':isABS,'isCMF':isCMF, 'isCDX':isCDX
            })
        return curve_data

    def get_swap_yield_quotes(self, country_code, trade_start_date, trade_end_date):
        swap_yield_sql = self.curve_queries.derived_swap_yield(country_code, trade_start_date, trade_end_date)
        df = self.curve_queries.fetch_mac_curves_query_dataframe(swap_yield_sql)
        columns = df.columns
        df_swap_yield = df[columns[2:]].groupby(['TradeDate', 'InYears']).sum()
        return df_swap_yield

    def get_irswap_quotes(self, country_code, currency_code, trade_start_date, trade_end_date):
        euro_country_codes = 'AT|BE|CY|DE|DK|EE|EP|ES|FI|FR|GR|IE|IT|LT|LU|MT|NL|PT|SI|SE|SK'
        # all countries in EUR currency get assigned one swap curve -> EU.EUR.IRSWP.ZC
        if country_code in euro_country_codes and currency_code == 'EUR':
            country_code = 'EU'
        parent_curve_short_name = country_code+'.'+currency_code+'.IRSWP.ZC'

        result_dict = CurveGenerator.curves_cache.get(parent_curve_short_name, {})
        df = pd.DataFrame()

        if not len(result_dict):
            result_dict = self.run_curve(parent_curve_short_name, trade_start_date, trade_end_date)
            if not len(result_dict):
                return df

        for trade_date, result in result_dict.items():
            df = df.append(pd.DataFrame(result))
        df.columns = ['TradeDate','Quote','InYears']
        df['CurveShortName'] = parent_curve_short_name

        return df

    def calculate_swap_spreads(self, curve_short_name, country_code, currency_code, curve_id, trade_start_date,
                               trade_end_date):
        """ Retrieve IR Swap yields and government yields, compute and store the difference
        """
        curve_data = {}
        CurveGenerator.curves_metadata.setdefault(curve_short_name, {}).update({'parent_curve_short_name': None})

        # Retrieve derived ir swap yield curve


        df_swap_quotes = self.get_irswap_quotes(country_code, currency_code, trade_start_date, trade_end_date)

        if len(df_swap_quotes) == 0:  # no swap curves available from start_date to end_date
            logging.warning('No swap quotes for:' + curve_short_name + '->' + trade_start_date + ':' + trade_end_date)
            return curve_data

        # Retrieve derived government yield curve
        df_gvt_quotes = self.get_sovereign_zero_curve_node_quotes_by_country(country_code, currency_code, trade_start_date, trade_end_date)

        #print gvt_query
        if len(df_gvt_quotes) == 0:  # no swap curves available from start_date to end_date
            logging.warning('No gvt (sovereign) quotes for:' + curve_short_name + '->' + trade_start_date + ':' + trade_end_date)
            return curve_data

        if len(df_gvt_quotes.CurveShortName.unique()) > 1:
            logging.warning('More than one government curve found for:' + curve_short_name + '->' + trade_start_date + ':' + trade_end_date)
            return curve_data

        curve_name_gvt = df_gvt_quotes.CurveShortName.unique().take(0)

        if len(df_swap_quotes.CurveShortName.unique()) > 1:
            logging.warning('More than one swap curve found for:' + curve_short_name + '->' + trade_start_date + ':' + trade_end_date)
            return curve_data

        curve_name_swp = df_swap_quotes.CurveShortName.unique().take(0)
        curve_name_swp_spr = curve_name_gvt.replace('GVT','SWP') + 'S'  # convert curve name to swap spread name

        df_comb = df_swap_quotes.append(df_gvt_quotes)
        fields = ['TradeDate', 'InYears', 'CurveShortName','Quote']
        df_swp_spr = df_comb[fields].set_index(fields[:-1]).unstack()['Quote']
        df_swp_spr[curve_name_swp_spr] = df_swp_spr[curve_name_swp] - df_swp_spr[curve_name_gvt]

        df_swp_spr.reset_index(inplace=True)

        # Retrieve node details
        sql_curve_nodes = self.curve_queries.curve_nodes(curve_id)
        df_curve_nodes = self.curve_queries.fetch_mac_curves_query_dataframe(sql_curve_nodes)
        if len(df_curve_nodes) == 0:
            logging.warning('No curve nodes found for:'
                            + curve_short_name + '->' + trade_start_date + ':' + trade_end_date)
            return curve_data

        # print df_swap_quotes
        # compute swap spread for each TradeDate
        for trade_date, df_swp_spr_curve in df_swp_spr.groupby('TradeDate'):
            if len(df_swp_spr_curve[curve_name_swp].dropna()) == 0:  # there are no swap quotes for trade date
                logging.info('No values in swap yield curve: ' + curve_name_swp + ' for trade date:' + trade_date)
                continue

            if len(df_swp_spr_curve[curve_name_gvt].dropna()) == 0:  # there are no swap quotes for trade date
                logging.info('No values in gvt curve '+ curve_name_gvt + ' for trade date:' + trade_date)
                continue

            df_swp_spr_curve_sorted = df_swp_spr_curve.sort('InYears') # Must sort by tenor for interp function to work
            tenors = sorted(df_curve_nodes['InYears'])

            # handle nan values on the gvt curve - the tenor did not match swap yield curve
            gvt_nan_index = np.isnan(df_swp_spr_curve_sorted[curve_name_gvt])
            gvt_nan_count = len(gvt_nan_index[gvt_nan_index])
            if gvt_nan_count > 0:
                gvt_nans = [str(val) for val in sorted(list(set(df_swp_spr_curve_sorted[gvt_nan_index]['InYears'])))]
                if (gvt_nan_count == 1) and (df_swp_spr_curve_sorted[gvt_nan_index]['InYears'] < 1.0/250.0).any():
                    # proceed with curve building because missing value is on the end point (short-end) of the curve
                    logging.warning(curve_name_gvt + ':'+ trade_date + ' nodes do not align with swap spread. Missing Tenor:' + ', '.join(gvt_nans))
                    df_swp_spr_curve.dropna(inplace=True)
                else:
                    # TODO: instead of dropping nans linearly interpolate values (assuming nan does not occur at first or last tenor)
                    logging.warning(curve_name_swp_spr + ':' + trade_date + ': ' + str(gvt_nan_count) + ' tenors misaligned. Cannot generate spread curve. Tenors:' + ', '.join(gvt_nans))
                    continue

            # Interpolate Swap Yield for all tenors
            curve_node_values = interp(tenors, df_swp_spr_curve['InYears'], df_swp_spr_curve[curve_name_swp_spr])

            curve_nodes=[{'date': trade_date, 'tenor': tenor, 'quotes': round(quote, 8)} for tenor, quote in
                        zip(tenors, curve_node_values)]

            has_nans = len(filter(lambda x: np.isnan(x['quotes']), curve_nodes)) > 0

            if has_nans:
                logging.warning('Nan interpolated values found for trade date: ' + trade_date + '. cannot build curve')
                continue

            if not has_nans:
                # store curve nodes in curve_data dictionary
                curve_data.setdefault(trade_date, curve_nodes)

        if len(curve_data):

            CurveGenerator.curves_cache.setdefault(curve_short_name, {}).update(curve_data)
        return curve_data

    def get_swap_spread_quotes_v2(self, curve_short_name, country_code, currency_code, curve_id, trade_start_date, trade_end_date, ric):
        curve_data = {}
        CurveGenerator.curves_metadata.setdefault(curve_short_name, {}).update({'parent_curve_short_name': None})

        #  Retrieve Reuters swap yield curve
        reuters_swap_query = self.curve_queries.swap_quotes(ric, country_code, currency_code, trade_start_date, trade_end_date)
        #print reuters_swap_query
        df_swap_quotes = self.curve_queries.fetch_mac_curves_query_dataframe(reuters_swap_query)
        if len(df_swap_quotes) == 0:  # no swap curves available from start_date to end_date
            logging.warning('No swap quotes for:' + curve_short_name + '->' + trade_start_date + ':' + trade_end_date)
            return curve_data

        df_swap_quotes = df_swap_quotes[['TradeDate', 'InYears', 'Quote']]
        df_swap_quotes['Quote'] /= 100.0  # convert Reuters yield to absolute value
        tenor_digits = 12  # driven from InYears column in tenor enum table so joins can be done on tenor values
        df_swap_quotes['InYears'] = df_swap_quotes['InYears'].round(tenor_digits)
        df_swap_quotes['CurveShortName'] = 'ReutersSwap'

        #  Retrieve underlying sovereign curve
        df_sovereign_quotes = self.get_sovereign_zero_curve_node_quotes_by_country(
            country_code,
            currency_code,
            trade_start_date,
            trade_end_date)
        df_sovereign_quotes['CurveShortName'] = 'SovYield'
        df_sovereign_quotes['InYears'] = df_sovereign_quotes['InYears'].round(tenor_digits)  # round tenor

        if len(df_sovereign_quotes) == 0:  # no sovereign curves available from start_date to end_date
            logging.warning('No sovereign quotes for:' + curve_short_name +
                            '->' + trade_start_date + ':' + trade_end_date)
            return curve_data

        #  combine swap yield and sovereign yield to one data frame to simplify processing
        df_swap_sov = df_sovereign_quotes.append(df_swap_quotes)

        #  pivot for processing by trade date (axis=1)
        df_pivot = df_swap_sov.groupby(['TradeDate', 'InYears', 'CurveShortName']).sum().unstack()
        df_pivot = df_pivot.reset_index()
        df_pivot.columns = ['TradeDate', 'InYears', 'ReutersSwap', 'SovYield']

        # compute swap spread for each TradeDate
        for trade_date, df in df_pivot.groupby('TradeDate'):
            trade_date_slice = df_pivot.TradeDate == trade_date
            tenors = df_pivot.loc[df_pivot.TradeDate == trade_date, 'InYears']

            #  Interpolate Sov Yield for all tenors
            sov_curve = df[['InYears', 'SovYield']].dropna()
            if len(sov_curve) == 0:  # there are no sov quotes for trade date
                logging.info('No sovereign curve for trade date:' + trade_date)
                continue
            df_pivot.loc[trade_date_slice, 'SovYieldInterp'] = interp(tenors, sov_curve['InYears'],
                                                                      sov_curve['SovYield'])

            #  Interpolate Swap Yield for all tenors
            swp_curve = df[['InYears', 'ReutersSwap']].dropna()
            if len(swp_curve) == 0:  # there are no swap quotes for trade date
                logging.info('No reuters swap curve for trade date:' + trade_date)
                continue
            df_pivot.loc[trade_date_slice, 'ReutersSwapInterp'] = interp(tenors, swp_curve['InYears'],
                                                                         swp_curve['ReutersSwap'])

            #  Compute Swap Spread for all tenors
            df_pivot.loc[trade_date_slice, 'SwapSpread'] = df_pivot.loc[trade_date_slice, 'ReutersSwapInterp'] - \
                df_pivot.loc[trade_date_slice, 'SovYieldInterp']

        df_curve_quotes = df_pivot[['TradeDate', 'InYears', 'SwapSpread']].dropna()
        df_curve_quotes.columns = ['date', 'tenor', 'quotes']
        curve_quotes = df_curve_quotes.T.to_dict().values()

        for date, group in groupby(sorted(curve_quotes, key=lambda x: x['date']), lambda x: x['date']):
            curve_data.setdefault(date, list(group))

        if len(curve_data):
            CurveGenerator.curves_cache.setdefault(curve_short_name, {}).update(curve_data)
        return curve_data

    def get_irswap_yield_quotes(self, curve_short_name, country_code, currency_code, curve_id, trade_start_date,
                                trade_end_date, ric):
        """ Computes ir swap yield curve
        """
        curve_data = self.compute_swap_yields(curve_short_name, country_code, currency_code,
                                              curve_id, trade_start_date, trade_end_date,
                                              partial(self.curve_queries.swap_quotes, ric))
        return curve_data

    def get_oiswap_yield_quotes(self, curve_short_name, country_code, currency_code, curve_id, trade_start_date,
                                trade_end_date, ric):
        """ Computes overnight index swap yield curve
        """
        curve_data = self.compute_swap_yields(curve_short_name, country_code, currency_code,
                                              curve_id, trade_start_date, trade_end_date,
                                              partial(self.curve_queries.ois_swap_quotes, ric))
        return curve_data


    def get_fdswap_yield_quotes(self, curve_short_name, country_code, currency_code, curve_id, trade_start_date,
                                trade_end_date, fixing_tenor):
        """ Computes forwarding swap yield curve
        """
        curve_data = {}
        try:
            curve_data = self.compute_swap_yields(curve_short_name, country_code, currency_code, curve_id, trade_start_date,
                                              trade_end_date,
                                              partial(self.curve_queries.fwd_swap_quotes, fixing_tenor))
        except Exception as e:
            logging.warning(e.message)
        return curve_data

#
    def compute_swap_yields(self, curve_short_name, country_code, currency_code,
                            curve_id, trade_start_date, trade_end_date, query_func):
        """ Core method for deriving swap yield curves (Forwarding, Single Curve, OIS, Basis Swaps
        """
        curve_data = {}
        CurveGenerator.curves_metadata.setdefault(curve_short_name, {}).update({'parent_curve_short_name': None})

        # Retrieve Reuters swap yield curve
        reuters_swap_query = query_func(country_code, currency_code, trade_start_date, trade_end_date)
        #print reuters_swap_query
        df_swap_quotes = self.curve_queries.fetch_mac_curves_query_dataframe(reuters_swap_query)
        if len(df_swap_quotes) == 0:  # no swap curves available from start_date to end_date
            err_msg = "No swap quotes for: {curve_short_name} -> {trade_start_date} : {trade_end_date}".format(curve_short_name=curve_short_name, 
                    trade_start_date = trade_start_date, trade_end_date=trade_end_date)
            err_msg += "\n Please check if today is a holiday. Fixed income calender is not implemented yet.\n"
            err_msg += "Query: %s"%(reuters_swap_query)
            if country_code not in ('ID', 'IN', 'HR'):
                raise Exception(err_msg)
            else:
                logging.warning(err_msg)
            return curve_data

        # Retrieve node details
        sql_curve_nodes = self.curve_queries.curve_nodes(curve_id)
        df_curve_nodes = self.curve_queries.fetch_mac_curves_query_dataframe(sql_curve_nodes)
        if len(df_curve_nodes) == 0:
            logging.warning('No curve nodes found for:' + curve_short_name + '->' + trade_start_date + ':' + trade_end_date)
            return curve_data

        # Reorder columns for pretty printing
        df_swap_quotes = df_swap_quotes[['TradeDate', 'InYears', 'Quote','DiscFactor']]
        df_swap_quotes['Quote'] /= 100.0  # convert Reuters yield to absolute value
        tenor_digits = 12  # driven from InYears column in tenor enum table so joins can be done on tenor values

        # Round nominal tenor
        df_swap_quotes['InYears'] = df_swap_quotes['InYears'].map(lambda x: round(x, tenor_digits)
        if x is not None else None )

        # Infer exact tenor from Discount Factor and Quote because Reuters is not correctly aligned
        df_swap_quotes['CalcTenor'] = -np.log(df_swap_quotes['DiscFactor']) / np.log(df_swap_quotes['Quote'] + 1.0)

        # convert to continuous compounding
        df_swap_quotes['Quote'] = np.log(1.0 + df_swap_quotes['Quote'])

        df_invalid_index = (df_swap_quotes['DiscFactor'] == 1) & (df_swap_quotes['Quote'] == 0.0)
        df_invalid = df_swap_quotes[df_invalid_index]
        if len(df_invalid) > 0:
            logging.info('Invalid tenor calculation encountered:')
            logging.info(df_invalid)
            df_swap_quotes['CalcTenor'].loc[df_invalid_index] = df_swap_quotes['InYears'][df_invalid_index]

        #print df_swap_quotes
        # compute swap spread for each TradeDate
        for trade_date, df_swp_curve in df_swap_quotes.groupby('TradeDate'):
            df_swp_curve.sort('InYears', inplace=True) # Must sort by tenor for interp function to work
            tenors = df_curve_nodes['InYears']
            #  Interpolate Swap Yield for all tenors
            if len(df_swp_curve) == 0:  # there are no swap quotes for trade date
                logging.info('No reuters swap curve for trade date:' + trade_date)
                continue

            curve_node_values = interp(tenors, df_swp_curve['CalcTenor'], df_swp_curve['Quote'])

            curve_nodes=[{'date': trade_date, 'tenor': tenor, 'quotes': round(quote, 8)} for tenor, quote in
                        zip(tenors, curve_node_values)]

            has_nans = len(filter(lambda x: np.isnan(x['quotes']), curve_nodes)) > 0
            if not has_nans:
                curve_data.setdefault(trade_date, curve_nodes)
            else:
                logging.info('Nan interpolated values found for trade date: ' + trade_date)

            if not has_nans:
                curve_data.setdefault(trade_date, curve_nodes)

        if len(curve_data):
            CurveGenerator.curves_cache.setdefault(curve_short_name, {}).update(curve_data)
        return curve_data


    def getBaseCorrelations(self, curve_short_name, curve_id, curve_long_name, trade_start_date, trade_end_date):
        date_range = [d.strftime('%Y-%m-%d') for d in pd.to_datetime(pd.date_range(start=trade_start_date, end=trade_end_date, freq='D'))]
        def forward_fill(d):
            tenor = d['tenor'].values[0]
            d.set_index('date', inplace=True)
            d = d.reindex(date_range)
            d['date'] = d.index
            d['tenor'] = tenor
            d.fillna(method='ffill', inplace=True)
            return d
        
        curve_data = {}
        query = self.curve_queries.base_correlation(trade_start_date, trade_end_date, curve_long_name.replace(" Base Correlation",""))
        correlation_results = pd.DataFrame([{'date':date, 'index_name':index_name, 'tenor':detachment*100.0, 'quotes':base_correlation} for date, index_name, red_code, detachment, base_correlation in self.curve_queries.fetch_mac_curves_query_result(query)])
        if len(correlation_results):    
            correlation_results = correlation_results.groupby(['index_name', 'tenor'], as_index=False, group_keys=False).apply(lambda x:forward_fill(x)).reset_index(drop=True).dropna(subset=['quotes'])
            cols = ['tenor', 'quotes', 'date']
            curve_quotes = correlation_results[cols].T.to_dict().values()
            for date, group in groupby(sorted(curve_quotes, key=lambda x: x['date']), lambda x: x['date']):
                curve_data.setdefault(date, list(group))
        if len(curve_data):
            CurveGenerator.curves_cache.setdefault(curve_short_name, {}).update(curve_data)
        return curve_data
    
    def get_constant_maturity_futures_quotes(self, curve_short_name, futures_contract_code, curve_id, currency_code, trade_start_date, trade_end_date):
        def _calculate_log_return(d):
            d['quotes'] = np.log(d['settlement_price']/d['settlement_price'].shift())
            return d

        def filter_bad_bonds(curve_id, curve_quotes):
            if len(curve_quotes):
                bad_bonds_data = pd.DataFrame([{'date':trade_date, 'future_contract_code':instrcode, 'ItemId':itemid} for trade_date, instrcode, itemid in
                                                                                                                            self.curve_queries.fetch_mac_curves_query_result("""select TradeDate, InstrCode, ItemId
                                                                                                                            from MarketData.dbo.DerivCurveFilteredBondCorrection where CurveId=%d"""%(curve_id))])
                if len(bad_bonds_data):
                    recurring_bad_bonds = bad_bonds_data[bad_bonds_data['ItemId'] == 14]['future_contract_code'].values
                    curve_quotes = curve_quotes[curve_quotes['future_contract_code'].map(lambda x:x not in recurring_bad_bonds)]
                    dated_bad_bonds = bad_bonds_data[bad_bonds_data['ItemId'] == 1]
                    if len(dated_bad_bonds):
                        merged_df = pd.merge(curve_quotes, dated_bad_bonds, how='left', left_on = ['future_contract_code', 'date'], right_on = ['future_contract_code', 'date'])
                        filtered_df = merged_df[pd.isnull(merged_df['ItemId'])]
                        return filtered_df
            return curve_quotes

        cols = ['date', 'tenor', 'quotes']
        _input_contracts = {}
        curve_data = {}
        if futures_contract_code is None:
            logging.info("Curve %s was not generated. Futures contract code is NULL"%curve_short_name)
            return curve_data
        tenor_enum = {k:v for k,v in self.curve_queries.fetch_mac_curves_query_result('select TenorEnum, InYears from [MarketData].[dbo].[TenorEnum]')}
        query = r"""select MAX(Date_) as MaxDate from qai.dbo.DSFutContrVal
                    join qai.dbo.DSFutContrInfo on qai.dbo.DSFutContrInfo.FutCode=qai.dbo.DSFutContrVal.FutCode
                    where ContrCode = %d and Date_ < '%s' and Settlement is not null and LastTrdDate is not null
                  """%(futures_contract_code, trade_start_date)

        relative_start_date = self.curve_queries.fetch_mac_curves_query_result(query)[0][0]
        if relative_start_date:
            relative_start_date = relative_start_date.strftime('%Y-%m-%d')
        else:
            relative_start_date = trade_start_date

        curve_quotes = [{'date':date, 'settlement_price':settlement_price,'last_trade_date':last_trade_date, 'future_contract_code':future_contract_code} 
                           for date, settlement_price, future_contract_code, last_trade_date in
                           self.curve_queries.fetch_mac_curves_query_result(self.curve_queries.futures_price_contract_code(futures_contract_code, relative_start_date, trade_end_date))
                           if settlement_price > 0.0
                        ]

        if len(curve_quotes):
            curve_quotes = pd.DataFrame(sorted(sorted(curve_quotes, key=lambda x:x['date']), key=lambda x:x['future_contract_code']))
            curve_quotes = curve_quotes.groupby('future_contract_code', as_index=False, group_keys=False).apply(lambda x:_calculate_log_return(x))
            curve_quotes = curve_quotes.dropna(subset=['quotes'])
            #check if country_code is US
            #if yes then remove US holidays from curve_quotes
            if currency_code == 'USD':
                holiday_dates = [result[0] for result in self.curve_queries.fetch_market_query_result("select DT from marketdb_global.META_HOLIDAY_CALENDAR_ACTIVE where ISO_CTRY_CODE = 'US' order by DT")]
                curve_quotes = curve_quotes[curve_quotes['date'].apply(lambda x: x not in holiday_dates)]
            #filter out bad bonds
            curve_quotes = filter_bad_bonds(curve_id, curve_quotes)

            if len(curve_quotes):
                curve_quotes['last_trade_date'] = pd.to_datetime(curve_quotes['last_trade_date'])
                curve_quotes['date'] = pd.to_datetime(curve_quotes['date'])
                curve_quotes['tenor'] = curve_quotes.apply(lambda d:float((d['last_trade_date']-d['date']).days)/(365), axis=1)
                curve_quotes = curve_quotes.groupby('date', as_index=False, group_keys=False).apply(lambda d:d.sort('tenor', ascending=True)).reset_index(drop=True)
                curve_quotes['date'] = curve_quotes['date'].map(lambda x:x.strftime('%Y-%m-%d'))
                _input_contracts = curve_quotes.copy().T.to_dict().values()
                curve_quotes = curve_quotes[cols][curve_quotes['date']>=trade_start_date].T.to_dict().values()
                curve_quotes = sorted(curve_quotes, key=lambda x:x['date'])
                query = 'select cn.TenorEnum,cn.CurveNodeId from [MarketData].[dbo].[CurveNodes] as cn where cn.CurveId=%d'%(curve_id)
                curve_nodes = {tenor_enum[tenor]: curve_node for tenor, curve_node in self.curve_queries.fetch_mac_curves_query_result(query)}
                nan_list = []
                for date, group in groupby(curve_quotes, key=lambda x:x['date']):
                    data = list(group)
                    curve_quotes_tenor = [d['tenor'] for d in data]
                    for node_tenor in curve_nodes.keys():
                        if node_tenor not in curve_quotes_tenor:
                            nan_list.append({'date':date,'tenor':node_tenor,'quotes':np.nan})
                curve_quotes.extend(nan_list)
                curve_quotes = self.run_curve_interpolation(curve_quotes)
                for date, group in groupby(sorted(curve_quotes, key=lambda x:x['date']), lambda x:x['date']):
                    # grouping by date
                    curve_data.setdefault(date,list(group))
                for date, group in groupby(sorted(_input_contracts, key=lambda x:x['date']), lambda x:x['date']):
                    # grouping by date
                    CurveGenerator.curves_input_contracts.setdefault(curve_short_name,{}).setdefault(date,[]).extend(list(group))
                if len(curve_data):
                    CurveGenerator.curves_cache.setdefault(curve_short_name,{}).update(curve_data)
        return curve_data


    def compute_spread_over_benchmark(self, df_quotes, country_code, currency_code, trade_start_date, trade_end_date, swap_benchmark):
        if swap_benchmark:
            # swap curve is the benchmark so pull swap yield curve
            df_curve_benchmark = self.get_irswap_quotes(country_code, currency_code, trade_start_date, trade_end_date)
        else:
            # sovereign curve is the benchmark so pull sovereign yield curve
            df_curve_benchmark = self.get_sovereign_zero_curve_node_quotes_by_currency(country_code, currency_code,
                                                                                       trade_start_date, trade_end_date)

        if len(df_curve_benchmark) == 0:
            # Could not pull benchmark curves for trade dates
            err_msg = 'No benchmark curves available for trade date {} to {}'.format(trade_start_date, trade_end_date)
            raise Exception(err_msg)

        df_corp_quotes = df_quotes.copy(deep=True)
        df_curve_benchmark.sort('InYears', inplace=True)

        def create_benchmark_curve(trade_date):
            df_curve = df_curve_benchmark[df_curve_benchmark['TradeDate'] == trade_date]
            if len(df_curve) == 0:
                return lambda _: np.nan
            df_curve.sort('InYears', inplace=True)
            def benchmark_curve(tenor, curve=df_curve):
                # df_curve_benchmark has columns 'InYears' (the node tenor) and 'Quote' (the yield level)
                yield_at_tenor = interp(tenor, curve['InYears'], curve['Quote'])
                return yield_at_tenor
            return benchmark_curve

        benchmark_curves = {date: create_benchmark_curve(date) for date, df in df_corp_quotes.groupby('date')}
        df_corp_quotes['benchmark_yield'] = df_corp_quotes.apply(lambda x: benchmark_curves[x['date']](x['tenor']), axis=1)
        df_corp_quotes['corp_spread_over_bench'] = df_corp_quotes['MatCorpYld'] - df_corp_quotes['benchmark_yield']

        return df_corp_quotes

    def memoize_dataframe(self, query, processor_func, arg_tuple, replace=False):
        if not replace and query in self.corp_dataframe_cache:
            df_quotes = self.corp_dataframe_cache[query]
        else:
            df_quotes = processor_func(*arg_tuple)
            self.corp_dataframe_cache[query] = df_quotes.copy(deep=True)
        return df_quotes

    def process_and_calculate_rating(self, df, econ_date):
        df_quotes = self.process_quotes(df, econ_date)
        df_quotes = self.compute_rating_spread(df_quotes)
        return df_quotes

    def process_and_calculate_rating_sector_issuer(self, df, econ_date):
        df_quotes = self.process_quotes(df, econ_date)
        df_quotes = self.compute_rating_spread(df_quotes)
        df_quotes = self.compute_sector_spread(df_quotes)
        df_quotes = self.compute_issuer_spread(df_quotes)
        return df_quotes

    def process_quotes(self, df_corp_quotes, econ_date, instrument_id='InstrCode', swap_benchmark=True):
        count_before_dupes_and_nans = len(df_corp_quotes)
        df_corp_quotes = self.drop_dupes_and_nans(df_corp_quotes, instrument_identifier=instrument_id)
        count_after_dupes_and_nans = len(df_corp_quotes)
        df_corp_quotes = self.scale_yields(df_corp_quotes)
        df_corp_quotes = self.remove_yield_outliers(df_corp_quotes)
        count_after_remove_yield_outliers = len(df_corp_quotes)
        df_corp_quotes = self.process_dates(df_corp_quotes)
        # add 'corp_spread_over_bench' field
        df_corp_quotes = self.compute_spread_over_benchmark(df_corp_quotes,
                                                            econ_date.country_code,
                                                            econ_date.currency_code,
                                                            econ_date.start_date,
                                                            econ_date.end_date,
                                                            swap_benchmark)
        df_corp_quotes = self.remove_spread_outliers(df_corp_quotes)
        count_after_remove_spread_outliers = len(df_corp_quotes)
        return df_corp_quotes

    def remove_yield_outliers(self, df_quotes):
        # Filter out bad values : negative prices, negative yields, extreme yields, zero notional amounts
        df_quotes = df_quotes[df_quotes['Amt'] > 0.0]
        df_quotes = df_quotes[df_quotes['MatCorpYld'] > 0.0]
        df_quotes = df_quotes[df_quotes['MatCorpYld'] < 0.5]
        return df_quotes

    def remove_spread_outliers(self, df_quotes, std_multiplier=3.0):
        # Filter out spreads that exceed threshold based on cross-sectional standard devation
        df_quotes['amt_pre_clip'] = df_quotes.groupby('CompositeRatingEnum')['Amt'].transform(np.sum)
        df_quotes['rating_spread_weight_pre_clip'] = df_quotes['Amt']/df_quotes['amt_pre_clip']
        df_quotes['rating_spread_contrib_pre_clip'] = df_quotes['rating_spread_weight_pre_clip']*df_quotes['corp_spread_over_bench']
        df_quotes['rating_spread_pre_clip'] = df_quotes.groupby('CompositeRatingEnum')['rating_spread_contrib_pre_clip'].transform(np.sum)
        df_quotes['rating_spread_std_pre_clip'] = df_quotes.groupby('CompositeRatingEnum')['corp_spread_over_bench'].transform(np.std)
        df_quotes['clipped'] = np.abs(df_quotes['corp_spread_over_bench']-df_quotes['rating_spread_pre_clip']) >= std_multiplier*df_quotes['rating_spread_std_pre_clip']
        count_clipped = len(df_quotes[df_quotes['clipped'] == True])
        df_quotes = df_quotes[df_quotes['clipped'] == False]
        return df_quotes
    
    def process_dates(self, df_quotes):
        df_quotes['TradeDate'] = pd.to_datetime(df_quotes['TradeDate'])
        df_quotes['date'] = df_quotes['TradeDate'].map(lambda x: x.strftime('%Y-%m-%d'))
        df_quotes['MatDate'] =  pd.to_datetime(df_quotes['MatDate'])
        df_quotes['tenor'] = (df_quotes['MatDate']-df_quotes['TradeDate']).dt.days.astype(float)/365.25
        return df_quotes
    
    def scale_yields(self, df_quotes):
        df_quotes['MatCorpYld'] /= 100.0
        return df_quotes

    def drop_dupes_and_nans(self, df_quotes, instrument_identifier='InstrCode'):
        df_quotes = df_quotes.drop_duplicates(subset=['TradeDate', instrument_identifier])
        # WrstCorpYld is not used in calculation or filters  but is considered a first class measure for legacy reasons (C# code)
        df_quotes = df_quotes.dropna(subset=['MatCorpYld', 'MatDate', 'MatCorpYld', 'Amt', 'WrstCorpYld'])
        return df_quotes

    def get_currency_rating_quote(self, curve_short_name, df_corp_quotes, field_name='rating_spread'):
        curve_data = {}
        for date, df_corp_quotes_group in df_corp_quotes.groupby('date'):
            if not len(df_corp_quotes_group):
                logging.warning('curve did not generate for {} {}'.format(date, curve_short_name))
                continue

            amount_weighted_average = df_corp_quotes_group[field_name].values[0]
            #df_corp_quotes_date = self.compute_sector_spread(df_corp_quotes_date)
            #df_corp_quotes_date = self.compute_issuer_spread(df_corp_quotes_date)
            corporate_curves_quotes_group_list = list(df_corp_quotes_group.T.to_dict().values())
            curve_data.setdefault(date, {'amount_weighted_corporate_spread': amount_weighted_average,
                                         'corporate_curve_quotes': corporate_curves_quotes_group_list})
        if len(curve_data):
            CurveGenerator.curves_cache.setdefault(curve_short_name,{}).update(curve_data)
        else:
            logging.warning('curve did not generate for {} '.format(curve_short_name))
        return curve_data


    def compute_sector_spread(self, df_corp_quotes):
        fields_group = ['CompositeRatingEnum', 'GicsLevel1']
        df_corp_quotes['sector_amt'] = df_corp_quotes.groupby(fields_group)['Amt'].transform(np.sum)
        df_corp_quotes['sector_weight'] = df_corp_quotes['Amt']/df_corp_quotes['sector_amt']
        df_corp_quotes['sector_spread_over_rating_spread'] = df_corp_quotes['corp_spread_over_bench'] - df_corp_quotes['rating_spread']
        df_corp_quotes['sector_spread_contribution'] = df_corp_quotes['sector_spread_over_rating_spread']*df_corp_quotes['sector_weight']
        df_corp_quotes['sector_spread'] = df_corp_quotes.groupby(fields_group)['sector_spread_contribution'].transform(np.sum)

        return df_corp_quotes

    def compute_issuer_spread(self, df_corp_quotes):
        fields_group = ['UltParIsrId']
        df_corp_quotes['UltParIsrId'] = df_corp_quotes['UltParIsrId'].fillna(0).astype(int)
        df_corp_quotes['issuer_amt'] = df_corp_quotes.groupby(fields_group)['Amt'].transform(np.sum)
        df_corp_quotes['issuer_weight'] = df_corp_quotes['Amt']/df_corp_quotes['issuer_amt']
        df_corp_quotes['issuer_spread_over_sector_spread'] = df_corp_quotes['corp_spread_over_bench'] - df_corp_quotes['rating_spread'] - df_corp_quotes['sector_spread']
        df_corp_quotes['issuer_spread_contribution'] = df_corp_quotes['issuer_spread_over_sector_spread']*df_corp_quotes['issuer_weight']
        df_corp_quotes['issuer_spread'] = df_corp_quotes.groupby(fields_group)['issuer_spread_contribution'].transform(np.sum)

        return df_corp_quotes

    def get_query_results(self, query, query_exec_func):
        logging.debug('running sql: \n' + query)
        df_quotes = query_exec_func(self.curve_queries, query)
        if df_quotes is None:
            raise Exception('failed query execution ' + query)
        quote_count = len(df_quotes)
        if quote_count == 0:
            raise Exception('No quotes returned from stored proc:\n' + query)
        logging.debug('database returned ' + str(quote_count) + ' quotes')
        return df_quotes

    def get_corp_ccy_quote(self, curve_short_name, econ_date, debt_spec, query_exec_func):

        query = self.curve_queries.derived_data_corp_bond_price_currency(econ_date.currency_code,
                                                                         econ_date.start_date,
                                                                         econ_date.end_date,
                                                                         debt_spec.issuer_id,
                                                                         debt_spec.iss_type_code,
                                                                         debt_spec.org_type_code,
                                                                         debt_spec.snrty_code)

        df_quotes = self.get_query_results(query, query_exec_func)

        return query, df_quotes

    def get_corp_abs_quote(self, curve_short_name, econ_date, debt_spec, query_exec_func):

        query = self.curve_queries.derived_data_corp_bond_price_abs(econ_date.currency_code,
                                                                    econ_date.start_date,
                                                                    econ_date.end_date,
                                                                    debt_spec.iss_type_code,
                                                                    debt_spec.top_composite_grade_incl,
                                                                    debt_spec.bottom_composite_grade_incl)
        df_quotes = self.get_query_results(query, query_exec_func)
        return query, df_quotes

    def process_corp_rating_sector_issuer(self, curve_short_name, econ_date, curve_spec, query_exec_func=extract_dataframe):
        debt_spec = DebtSpec(iss_type_code='CORP')
        query, df_quotes = self.get_corp_ccy_quote(curve_short_name, econ_date, debt_spec, query_exec_func)
        df_quotes = self.memoize_dataframe(query, self.process_and_calculate_rating_sector_issuer, (df_quotes, econ_date))
        return df_quotes

    def get_corp_rating_spread_quote(self, curve_short_name, econ_date, curve_spec, query_exec_func=extract_dataframe):
        df_quotes = self.process_corp_rating_sector_issuer(curve_short_name, econ_date, curve_spec, query_exec_func)
        df_quotes = df_quotes[df_quotes['CompositeRatingEnum'] == curve_spec.composite_rating]
        curve_data = self.get_currency_rating_quote(curve_short_name, df_quotes)
        return curve_data

    def get_corp_sector_spread_quote(self, curve_short_name, econ_date, curve_spec, query_exec_func=extract_dataframe):
        df_quotes = self.process_corp_rating_sector_issuer(curve_short_name, econ_date, curve_spec, query_exec_func)
        df_quotes = df_quotes[df_quotes['CompositeRatingEnum'] == curve_spec.composite_rating]
        df_quotes = df_quotes[df_quotes['GicsLevel1'] == curve_spec.sector]
        curve_data = self.get_currency_rating_quote(curve_short_name, df_quotes, 'sector_spread')
        return curve_data

    def get_corp_issuer_spread_quote(self, curve_short_name, econ_date, curve_spec, query_exec_func=extract_dataframe):
        df_quotes = self.process_corp_rating_sector_issuer(curve_short_name, econ_date, curve_spec, query_exec_func)
        df_quotes = df_quotes[df_quotes['UltParIsrId'] == curve_spec.issuer_id]
        curve_data = self.get_currency_rating_quote(curve_short_name, df_quotes, 'issuer_spread')
        return curve_data

    def get_CAD_provincial_curve_quotes(self, curve_short_name, econ_date, curve_spec, query_exec_func=extract_dataframe):
        debt_spec = DebtSpec(iss_type_code='OTHR', org_type_code='GVT', issuer_id=curve_spec.issuer_id, snrty_code='UN')
        # based on issuer
        query, df_quotes = self.get_corp_ccy_quote(curve_short_name, econ_date, debt_spec, query_exec_func)
        df_quotes = self.process_quotes(df_quotes, econ_date, swap_benchmark=False)
        df_quotes = self.compute_category_spread(df_quotes, category_name=curve_short_name)
        curve_data = self.get_currency_rating_quote(curve_short_name, df_quotes, field_name='category_spread')
        return curve_data

    def get_supranational_curve_quotes(self, curve_short_name, econ_date, curve_spec, query_exec_func=extract_dataframe):
        debt_spec = DebtSpec(iss_type_code='OTHR', org_type_code='SUPN')
        query, df_quotes = self.get_corp_ccy_quote(curve_short_name, econ_date, debt_spec, query_exec_func)
        df_quotes = self.process_quotes(df_quotes, econ_date)
        if '(IG)' in curve_short_name:
            df_quotes = df_quotes[df_quotes['CompositeRatingEnum'] != 'SUB-IG']
            df_quotes = self.compute_category_spread(df_quotes, category_name=curve_short_name)
            curve_data = self.get_currency_rating_quote(curve_short_name, df_quotes, field_name='category_spread')
        else:
            df_quotes = df_quotes[df_quotes['CompositeRatingEnum'] == curve_spec.composite_rating]
            df_quotes = self.compute_category_spread(df_quotes, category_name=curve_short_name)
            curve_data = self.get_currency_rating_quote(curve_short_name, df_quotes, field_name='category_spread')

        return curve_data

    def get_agency_curve_quotes(self, curve_short_name, econ_date, curve_spec, query_exec_func=extract_dataframe):
        debt_spec = DebtSpec(iss_type_code='AGNC', org_type_code='GVTOC', snrty_code='UN', issuer_id=curve_spec.issuer_id)
        query, df_quotes = self.get_corp_ccy_quote(curve_short_name, econ_date, debt_spec, query_exec_func)
        df_quotes = self.process_quotes(df_quotes, econ_date, swap_benchmark=False)
        df_quotes = self.compute_category_spread(df_quotes, category_name=curve_short_name)
        curve_data = self.get_currency_rating_quote(curve_short_name, df_quotes, field_name='category_spread')
        return curve_data

    def create_abs_debt_spec(self, curve_short_name, econ_date):
        iss_type_temp = curve_short_name.replace(econ_date.currency_code+'.', '').replace(econ_date.country_code+'.', '')
        iss_type_temp = iss_type_temp.replace('(IG).','').replace('ABS','')
        debt_spec = DebtSpec(iss_type_code=iss_type_temp, top_composite_grade_incl='1', bottom_composite_grade_incl='10')
        return debt_spec

    def get_abs_curve_quotes(self, curve_short_name, econ_date, query_exec_func=extract_dataframe):
        debt_spec = self.create_abs_debt_spec(curve_short_name, econ_date)
        query, df_quotes = self.get_corp_abs_quote(curve_short_name, econ_date, debt_spec, query_exec_func)
        df_quotes = self.process_quotes(df_quotes, econ_date, instrument_id='Isin')
        df_quotes = self.compute_category_spread(df_quotes, category_name=curve_short_name)
        curve_data = self.get_currency_rating_quote(curve_short_name, df_quotes, field_name='category_spread')
        return curve_data

    def get_covered_bond_quotes(self, curve_short_name, country_code, currency_code, composite_rating, market_status_enum, trade_start_date, trade_end_date, curve_id):
        snrtyCode = "MTG|SEC|SRSEC"
        BktVectorBegin  = '0.5|1.5|3.5|7.5|20'
        BktVectorEnd    = '1.5|3.5|7.5|20|100'
        BktVectorTenor  = '1|2|5|10|30'
        debtIssTypeCountryCodeMap = {
                                'SE':{'regular':'CVDBND|SAKOBL'},#Sweden
                                'DE':{'regular':'CVDBND|PFANLEIH|PFHYPOTK|PFLAND|PFSCHIFF',
                                      'JUMBO':'FUSCHJU|HPF|JBLDSCHA|JMBFLUG|JMBLAND|JMBSCHIF|OPF'
                                     },#Germany
                                'ES':{'regular':'CVDBND|HF|CEDHIP'},#Spanish
                                'FR':{'regular':'CVDBND|FONCIER'},#France
                                'DK':{'regular':'CVDBND|REALKROB'}#Denmark
                             }
        tenor_enum = {k: v for k, v in self.curve_queries.fetch_mac_curves_query_result('select TenorEnum, InYears from [MarketData].[dbo].[TenorEnum]')}
        curve_nodes = self.curve_queries.fetch_mac_curves_query_result('select cn.TenorEnum,cn.CurveNodeId from [MarketData].[dbo].[CurveNodes] as cn where cn.CurveId=%d'%(curve_id))
        _curve_short_name = self.curve_queries.fetch_mac_curves_query_result(self.curve_queries.curve_detail_by_curvetypes_country(['SwapZrSpr'],'EP', trade_start_date, trade_end_date))[0][4]
        debt_iss_type_code = debtIssTypeCountryCodeMap[country_code][market_status_enum]\
                                if debtIssTypeCountryCodeMap[country_code].has_key(market_status_enum)\
                                else debtIssTypeCountryCodeMap[country_code]['regular']
        query = self.curve_queries.derived_data_corp_bond_term_structure(currency_code, country_code, trade_start_date, trade_end_date, debt_iss_type_code, snrtyCode, BktVectorBegin, BktVectorEnd, BktVectorTenor)
        corporate_curves_quotes = self.curve_queries.fetch_mac_curves_query_result(query)
        corporate_curves_quotes = [{'InstrCode': data[1],
                                    'ISIN': data[22],
                                    'date': data[15].strftime('%Y-%m-%d'),
                                    'tenor': 1.0*((data[4]-data[15]).days)/365.25,
                                    'Price': data[16],
                                    'WrstCorpYld': replace_null(data[18])/100.0,
                                    'MatCorpYld': replace_null(data[19])/100.0,
                                    'Amt': data[21],
                                    'CompositeRatingEnum': data[28],
                                    'AvgTTMByCurTerm': data[46],
                                    'AvgYldByCurTerm': replace_null(data[45])/100.0}
                                   for data in corporate_curves_quotes]

        corporate_curves_quotes = [data for data in corporate_curves_quotes if data['CompositeRatingEnum'] == composite_rating]
        corporate_curves_quotes = sorted(corporate_curves_quotes, key=lambda x:x['date'])
        number_of_quotes_total = len(corporate_curves_quotes)
        curve_data = {}
        if number_of_quotes_total is 0:
            warning_msg = "No bond quotes available for trade date {} to {}. Cannot build curve: {}" \
                .format(trade_start_date, trade_start_date, curve_short_name)
            logging.warning(warning_msg)
            return curve_data
        corporate_curves_quotes_df = pd.DataFrame(corporate_curves_quotes)
        corporate_curves_quotes_df.dropna(subset=['AvgYldByCurTerm', 'Amt', 'AvgTTMByCurTerm'], inplace=True)  # drop empty rows
        corporate_curves_quotes_df.drop_duplicates(subset=['date', 'InstrCode'], inplace=True)
        corporate_curves_quotes = list(corporate_curves_quotes_df.T.to_dict().values())

        for date, corp_quotes_grp_df in corporate_curves_quotes_df.groupby('date'):
            try:
                corp_quotes_grp_df.sort('AvgTTMByCurTerm', inplace=True)
                df_swap_curve = self.get_irswap_quotes(country_code, currency_code, trade_start_date, trade_end_date)
                if len(df_swap_curve) is 0:
                    warning_msg = "No swap curve found on trade date {}. Cannot build curve: {}"\
                        .format(date, curve_short_name)
                    logging.warning(warning_msg)
                    continue
                df_swap_curve.sort('InYears', inplace=True)
                def swap_curve_interp(tenor):
                    result = interp(tenor, df_swap_curve.InYears, df_swap_curve.Quote)
                    return result

                corp_quotes_grp_df['SwpYld'] = corp_quotes_grp_df.apply(lambda x: swap_curve_interp(x['AvgTTMByCurTerm']), axis=1)
                corp_quotes_grp_df['Spread'] = corp_quotes_grp_df['AvgYldByCurTerm'] - corp_quotes_grp_df['SwpYld']

                def covered_spread_interp(tenor):
                    corp_yield = interp(tenor, corp_quotes_grp_df['AvgTTMByCurTerm'], corp_quotes_grp_df['AvgYldByCurTerm'])
                    swap_yield = swap_curve_interp(tenor)
                    covered_spread = corp_yield - swap_yield
                    return covered_spread

                corporate_curves_quotes_date_list = list(corp_quotes_grp_df.T.to_dict().values())
                covered_bond_curve_node_tenors = sorted([tenor_enum[data[0]] for data in curve_nodes])
                covered_spread_dict = {x: covered_spread_interp(x) for x in covered_bond_curve_node_tenors}

                curve_data.setdefault(date,{'corporate_yield':corporate_curves_quotes_date_list,
                                             'covered_spread':covered_spread_dict,
                                             'corporate_curve_quotes':corporate_curves_quotes
                                              })
            except:
                logging.info(traceback.format_exc())
        if len(curve_data):
            logging.info("Loading covered bond derived data into in-memory cache...")
            CurveGenerator.curves_cache.setdefault(curve_short_name,{}).update(curve_data)
        else:
            logging.info("No derived curve was generated.")
        return curve_data


    def get_sovereign_spread_quotes(self, curve_short_name, currency_code, country_iso, trade_start_date, trade_end_date):
        start_date = date_parser.parse(trade_start_date)
        end_date = date_parser.parse(trade_end_date)
        final_curve_quotes = {}
        date_range = [date_value for date_value in self.calender_days_between(start_date, end_date)]
        #print date_range
        for current_date in date_range:
            curr_date_str = current_date.strftime('%Y-%m-%d')
            benchmark_query = self.curve_queries.derived_data_sovereign_curve_yields(curr_date_str, currency_code, 0, 1)
            #print benchmark_query
            benchmark_curve_quotes = self.curve_queries.fetch_mac_curves_query_result(benchmark_query)
            sovereign_sql_query = self.curve_queries.derived_data_sovereign_curve_yields(curr_date_str, currency_code, 1, None)
            #print sovereign_sql_query
            curve_quotes = self.curve_queries.fetch_mac_curves_query_result(sovereign_sql_query)
            #curve_quotes = sorted(curve_quotes, key=lambda x: x[2]) # sort on remaining term - interp requires proper sort
            benchmark_curve_quotes = [{'CompFreqCode':item[0], 'NomTermToMat':item[1], 'MatDate':item[2], 'Price':item[3],'tenor':1.0*(item[2]-current_date).days/365.25, 'WrstStdYld':replace_null(item[4])/100.0, 'WrstCorpYld':replace_null(item[5])/100.0, 'MatStdYld':replace_null(item[6])/100.0, 'MatCorpYld':replace_null(item[7])/100.0, 'InstrCode':item[8], 'IssName':item[9], 'IsoCtryCode':item[10]} for item in benchmark_curve_quotes]
            curve_quotes = [{'CompFreqCode':item[0], 'NomTermToMat':item[1], 'MatDate':item[2], 'Price':item[3], 'tenor':1.0*(item[2]-current_date).days/365.25, 'WrstStdYld':replace_null(item[4])/100.0, 'WrstCorpYld':replace_null(item[5])/100.0, 'MatStdYld':replace_null(item[6])/100.0, 'MatCorpYld':replace_null(item[7])/100.0, 'InstrCode':item[8], 'IssName':item[9], 'IsoCtryCode':item[10]} for item in curve_quotes]
            #print current_date, 'Benchmark Count: ', len(benchmark_curve_quotes)
            if len(benchmark_curve_quotes)==0:
                continue
            _data = [item for item in curve_quotes if item['IsoCtryCode'] == country_iso]
            if len(_data)==0:
                continue
            # before passing to interp - sort on remaining term because (x, y) needs to be in ascending order
            curve_quotes = pd.DataFrame(_data).drop_duplicates(subset='InstrCode').T.to_dict().values()
            curve_quotes = sorted(curve_quotes, key=lambda x: x['tenor'])
            benchmark_curve_quotes = pd.DataFrame(benchmark_curve_quotes).drop_duplicates(subset='InstrCode').T.to_dict().values()
            benchmark_curve_quotes = sorted(benchmark_curve_quotes, key=lambda x: x['tenor'])
            sovereign_tenors = [data['tenor'] for data in curve_quotes]
            sovereign_yields = [data['MatCorpYld'] for data in curve_quotes]
            benchmark_tenors = [data['tenor'] for data in benchmark_curve_quotes]
            benchmark_yields = [data['MatCorpYld'] for data in benchmark_curve_quotes]
            # to compute sovereign spreads we need to first interpolate benchmark yields to match
            # tenors for sovereign issues
            benchmark_yields_interp = interp(sovereign_tenors, benchmark_tenors, benchmark_yields)
            sovereign_spreads = [sov_yield - bench_yield for bench_yield, sov_yield
                                  in zip(benchmark_yields_interp, sovereign_yields)]
            sov_spr_5yr = interp(5.0, sovereign_tenors, sovereign_spreads)
            final_curve_quotes[curr_date_str] = {'corporate_curve_quotes':curve_quotes, 'sovereign_spread':sov_spr_5yr}
        if len(final_curve_quotes):
            CurveGenerator.curves_cache.setdefault(curve_short_name,{}).update(final_curve_quotes)
        return final_curve_quotes

    def get_sovereign_spread_quotes_emg(self, curve_short_name, currency_code, country_iso, trade_start_date, trade_end_date):
        current_date = date_parser.parse(trade_start_date)
        end_date = date_parser.parse(trade_end_date)
        final_curve_quotes = {}
        while current_date <= end_date:
            trade_date = current_date.strftime('%Y-%m-%d')
            query_sov_quotes = self.curve_queries.derived_data_sovereign_curve_yields(trade_date, currency_code, 0, 1)
            benchmark_curve_quotes = self.curve_queries.fetch_mac_curves_query_result(query_sov_quotes)
            #print self.curve_queries.derived_data_sovereign_curve_yields(current_date.strftime('%Y-%m-%d'), currency_code, 1, None)
            curve_quotes = self.curve_queries.fetch_mac_curves_query_result(self.curve_queries.derived_data_sovereign_curve_yields(current_date.strftime('%Y-%m-%d'), currency_code, 1, None))
            benchmark_curve_quotes = [{'CompFreqCode': item[0],
                                       'NomTermToMat':item[1],
                                       'MatDate': item[2],
                                       'Price': item[3],
                                       'tenor': 1.0*(item[2]-current_date).days/365.25,
                                       'WrstStdYld': replace_null(item[4])/100.0,
                                       'WrstCorpYld' :replace_null(item[5])/100.0,
                                       'MatStdYld': replace_null(item[6])/100.0,
                                       'MatCorpYld': replace_null(item[7])/100.0,
                                       'InstrCode': item[8],
                                       'IssName': item[9],
                                       'IsoCtryCode': item[10]}
                                      for item in benchmark_curve_quotes]

            curve_quotes = [{'CompFreqCode': item[0],
                             'NomTermToMat': item[1],
                             'MatDate': item[2],
                             'Price': item[3],
                             'tenor': 1.0*(item[2]-current_date).days/365.25,
                             'WrstStdYld': replace_null(item[4])/100.0,
                             'WrstCorpYld': replace_null(item[5])/100.0,
                             'MatStdYld': replace_null(item[6])/100.0,
                             'MatCorpYld': replace_null(item[7])/100.0,
                             'InstrCode': item[8],
                             'IssName': item[9],
                             'IsoCtryCode': item[10],
                             'Amt': item[11]}
                            for item in curve_quotes]

            if len(benchmark_curve_quotes):
                _data = [item for item in curve_quotes if item['IsoCtryCode'] == country_iso]
                if len(_data) > 1:
                    curve_quotes = pd.DataFrame(_data).drop_duplicates(subset='InstrCode').T.to_dict().values()
                    benchmark_curve_quotes = pd.DataFrame(benchmark_curve_quotes).drop_duplicates(subset='InstrCode').T.to_dict().values()
                    curve_quotes = sorted(curve_quotes, key=lambda x: x['tenor'])
                    benchmark_curve_quotes = sorted(benchmark_curve_quotes, key=lambda x: x['tenor'])
                    benchmark_curve_interp = 0
                    interpolated_data = zip([data['tenor'] for data in curve_quotes], interp([data['tenor'] for data in curve_quotes],[data['tenor'] for data in benchmark_curve_quotes], [data['MatCorpYld'] for data in benchmark_curve_quotes]))
                    for ic_data, c_data in zip(interpolated_data,curve_quotes):
                        assert c_data['tenor'] == ic_data[0]
                        c_data.update({'spread_quotes': c_data['MatCorpYld']-ic_data[1]})

                    df_curve_quotes = pd.DataFrame(curve_quotes)
                    df_curve_quotes.Amt = df_curve_quotes.Amt.astype(float).fillna(0.0)

                    if country_iso == 'AR':
                        df_short_end = df_curve_quotes[df_curve_quotes.tenor <= 7.0]
                    else:
                        df_short_end = df_curve_quotes[df_curve_quotes.tenor <= 5.5]
                    num_short_end = len(df_short_end)

                    if num_short_end > 1:  # if >1 bond with tenor <5 years: average bonds with tenor <5
                        # sum product dataframe style
                        sov_spr = (df_short_end.Amt*df_short_end.spread_quotes).sum()/df_short_end.Amt.sum()
                    else: # if <=1 bond with tenor <5 years: average all bonds
                        sov_spr = (df_curve_quotes.Amt*df_curve_quotes.spread_quotes).sum()/df_curve_quotes.Amt.sum()

                    final_curve_quotes[current_date.strftime('%Y-%m-%d')] = {'corporate_curve_quotes': curve_quotes,
                                                                             'sovereign_spread': sov_spr}
            current_date += datetime.timedelta(days=1)
        if len(final_curve_quotes):
            CurveGenerator.curves_cache.setdefault(curve_short_name,{}).update(final_curve_quotes)
        return final_curve_quotes


    def get_sovereign_spread_quotes_emg_CDS(self, curve_short_name, currency_code, country_iso, trade_start_date, trade_end_date):
        current_date = date_parser.parse(trade_start_date)
        end_date = date_parser.parse(trade_end_date)
        final_curve_quotes = {}
        while current_date <= end_date:
            curve_quotes = self.curve_queries.fetch_mac_curves_query_result(self.curve_queries.emg_CDS_spreads(current_date.strftime('%Y-%m-%d'), currency_code))
            #print self.curve_queries.derived_data_sovereign_curve_yields(current_date.strftime('%Y-%m-%d'), currency_code, 1, None)
            #curve_quotes = self.curve_queries.fetch_mac_curves_query_result(self.curve_queries.derived_data_sovereign_curve_yields(current_date.strftime('%Y-%m-%d'), currency_code, 1, None))
            #benchmark_curve_quotes = [{'CompFreqCode':item[0], 'NomTermToMat':item[1], 'MatDate':item[2], 'Price':item[3],'tenor':1.0*(item[2]-current_date).days/365.25, 'WrstStdYld':replace_null(item[4])/100.0, 'WrstCorpYld':replace_null(item[5])/100.0, 'MatStdYld':replace_null(item[6])/100.0, 'MatCorpYld':replace_null(item[7])/100.0, 'InstrCode':item[8], 'IssName':item[9], 'IsoCtryCode':item[10]} for item in benchmark_curve_quotes]

            curve_quotes = [{'TradeDate':item[0],'IsoCtryCode':item[1], 'IsoCrncyCode':item[2], '5Y':item[3]/100.0, '10Y':item[4]/100.0, '30Y':item[5]/100.0 } for item in curve_quotes]
            _data = [item for item in curve_quotes if item['IsoCtryCode'] == country_iso]
            if len(_data) > 0:
                curve_quotes = pd.DataFrame(_data).drop_duplicates().T.to_dict().values()
                sov_spr = curve_quotes['5Y']
                final_curve_quotes[current_date.strftime('%Y-%m-%d')] = {'corporate_curve_quotes':curve_quotes,
                                                                         'sovereign_spread':sov_spr}
            current_date += datetime.timedelta(days=1)
        if len(final_curve_quotes):
            CurveGenerator.curves_cache.setdefault(curve_short_name, {}).update(final_curve_quotes)
        return final_curve_quotes



    def run_curve_interpolation(self, curve_quotes):
        clean_quotes = []
        curve_quotes = sorted(curve_quotes, key=lambda x:x['date'])
        for date, date_group in groupby(curve_quotes, key=lambda x:x['date']): # group by date
            data = sorted(list(date_group), key=lambda x:x['tenor'])
            clean_data = [(record['tenor'],record['quotes'])for record in data if not np.isnan(record['quotes'])]
            bad_data = [record['tenor']for record in data if np.isnan(record['quotes'])]
            if len(clean_data)>0:
                interpolated_data = zip(bad_data,interp(bad_data,[d[0]for d in clean_data],[d[1]for d in clean_data]))
                for tenor, interpolated_quote in interpolated_data:
                    clean_data.append((tenor, interpolated_quote))
                for tenor, quotes in sorted(clean_data, key=lambda x:x[0]):
                    clean_quotes.append({'date':date,'tenor':tenor,'quotes':quotes})
        return clean_quotes
    
    def get_zero_curve_node_quotes(self, country_code, currency_code, trade_start_date, trade_end_date):
        if country_code is not None:
            if currency_code is not None:
                if currency_code == 'EUR':
                    country_code = 'EP'
            if country_code not in CurveGenerator.zero_curves_cache:
                results = self.get_zero_curve_node_quotes_by_country(country_code, currency_code, trade_start_date, trade_end_date)
                CurveGenerator.zero_curves_cache.setdefault(country_code, {}).update(results)
            return CurveGenerator.zero_curves_cache[country_code]
        else:
            assert currency_code is not None
            if currency_code not in CurveGenerator.zero_curves_cache:
                results = self.get_zero_curve_node_quotes_by_currency(currency_code, currency_code, trade_start_date, trade_end_date)
                CurveGenerator.zero_curves_cache.setdefault(currency_code, {}).update(results)
            return CurveGenerator.zero_curves_cache[currency_code]


    def get_zero_curve_node_quotes_by_country(self, country_code, currency_code, trade_start_date, trade_end_date):
        sov_zero_query = self.curve_queries.zero_curve_by_country(country_code, currency_code, trade_start_date, trade_end_date)
        curve_ids = self.curve_queries.fetch_mac_curves_query_result(sov_zero_query)
        curve_id = curve_ids[0][5]
        query = 'select cn.TenorEnum,cn.CurveNodeId ' \
                'from [MarketData].[dbo].[CurveNodes] as cn ' \
                'where cn.CurveId=%d' % curve_id
        curve_nodes = self.curve_queries.fetch_mac_curves_query_result(query)
        tenor_enum = {k: v for k, v in self.curve_queries.fetch_mac_curves_query_result(
            'select TenorEnum, InYears '
            'from [MarketData].[dbo].[TenorEnum]')}
        zero_curve_quotes = {}
        for (tenor, curve_node) in curve_nodes:
            sov_zero_quote_query = self.curve_queries.curve_node_quotes(curve_node, trade_start_date, trade_end_date)
            sov_zero_quote_results = self.curve_queries.fetch_mac_curves_query_result(sov_zero_quote_query)
            for quotes, date in sov_zero_quote_results:
                zero_curve_quotes.setdefault(date, {}).setdefault(tenor_enum[tenor], {'quotes':quotes})
        return zero_curve_quotes

    def get_sovereign_zero_curve_node_quotes_by_country(self, country_code, currency_code, trade_start_date, trade_end_date):
        curve_record_query_func = lambda _: self.curve_queries.zero_curve_by_country(country_code, currency_code)

        df_zero_quotes = self.get_sovereign_zero_curve_node_quotes(country_code, currency_code, trade_start_date,
                                                                   trade_end_date, curve_record_query_func)
        return df_zero_quotes

    def get_sovereign_zero_curve_node_quotes_by_currency(self, country_code, currency_code, trade_start_date, trade_end_date):
        if currency_code == 'EUR':
            country_code = 'EP'
        if currency_code == 'USD':
            country_code = 'US'
        curve_record_query_func = lambda _: self.curve_queries.zero_curve_by_currency(currency_code, country_code, trade_start_date, trade_end_date)
        df_zero_quotes = self.get_sovereign_zero_curve_node_quotes(country_code, currency_code, trade_start_date, trade_end_date, curve_record_query_func)
        return df_zero_quotes

    def get_sovereign_zero_curve_node_quotes(self, country_code, currency_code, trade_start_date, trade_end_date, query_func):
        sov_zero_curve_record_query = query_func(None)
        curve_record = self.curve_queries.fetch_mac_curves_query_result(sov_zero_curve_record_query)
        curve_short_name = curve_record[0][4]
        curve_query = self.curve_queries.curve_quotes_query_template(trade_start_date, trade_end_date, curve_short_name)
        df_zero_quotes = self.curve_queries.fetch_mac_curves_query_dataframe(curve_query)
        return df_zero_quotes
    
    def get_zero_curve_node_quotes_by_currency(self, currency_code, trade_start_date, trade_end_date):
        curve_id = self.curve_queries.fetch_mac_curves_query_result(
            self.curve_queries.zero_curve_by_currency(currency_code))[0][5]
        query = 'select cn.TenorEnum,cn.CurveNodeId ' \
                'from [MarketData].[dbo].[CurveNodes] as cn' \
                ' where cn.CurveId=%d' % curve_id
        curve_nodes = self.curve_queries.fetch_mac_curves_query_result(query)
        tenor_enum = {k: v for k, v in self.curve_queries.fetch_mac_curves_query_result(
            'select TenorEnum, InYears from [MarketData].[dbo].[TenorEnum]')}
        zero_curve_quotes = {}
        for (tenor, curve_node) in curve_nodes:
            for quotes, date in self.curve_queries.fetch_mac_curves_query_result(
                    self.curve_queries.curve_node_quotes(
                        curve_node, trade_start_date, trade_end_date)):
                zero_curve_quotes.setdefault(date,{}).setdefault(tenor_enum[tenor], {'quotes':quotes})
        return zero_curve_quotes
   
    def get_curve_node_quotes_by_curveid(self, curve_id, trade_start_date, trade_end_date):
        query = 'select cn.TenorEnum,cn.CurveNodeId from [MarketData].[dbo].[CurveNodes] as cn where cn.CurveId=%d'%(curve_id)
        curve_nodes = self.curve_queries.fetch_mac_curves_query_result(query)
        tenor_enum = {k: v for k, v in self.curve_queries.fetch_mac_curves_query_result('select TenorEnum, InYears from [MarketData].[dbo].[TenorEnum]')}
        curve_node_quotes = {}
        for (tenor, curve_node) in curve_nodes:
            for quotes, date in self.curve_queries.fetch_mac_curves_query_result(self.curve_queries.curve_node_quotes(curve_node, trade_start_date, trade_end_date)):
                curve_node_quotes.setdefault(date,{}).setdefault(tenor_enum[tenor], {'quotes':quotes})
        return curve_node_quotes

    def get_amount_weighted_corporate_spread(self, df_corp_curves_quotes):
        weights = df_corp_curves_quotes['Amt']/df_corp_curves_quotes['Amt'].sum()
        corporate_spreads = df_corp_curves_quotes['corp_spread_over_bench']
        return np.dot(weights, corporate_spreads)

    def compute_rating_spread(self, df_corp_quotes):
        fields = ['CompositeRatingEnum']
        df_corp_quotes = df_corp_quotes.copy(deep=True)
        df_corp_quotes['rating_amt'] = df_corp_quotes.groupby(fields)['Amt'].transform(np.sum)
        df_corp_quotes['rating_weight'] = df_corp_quotes['Amt']/df_corp_quotes['rating_amt']
        df_corp_quotes['rating_contrib'] = df_corp_quotes['rating_weight'] * df_corp_quotes['corp_spread_over_bench']
        df_corp_quotes['rating_spread'] = df_corp_quotes.groupby(fields)['rating_contrib'].transform(np.sum)
        return df_corp_quotes

    def compute_category_spread(self, df_corp_quotes, category_name='Category'):
        df_corp_quotes = df_corp_quotes.copy(deep=True)
        fields = ['category']
        df_corp_quotes['category'] = category_name
        df_corp_quotes['category_amt'] = df_corp_quotes.groupby(fields)['Amt'].transform(np.sum)
        df_corp_quotes['category_weight'] = df_corp_quotes['Amt']/df_corp_quotes['category_amt']
        df_corp_quotes['category_contrib'] = df_corp_quotes['category_weight'] * df_corp_quotes['corp_spread_over_bench']
        df_corp_quotes['category_spread'] = df_corp_quotes.groupby(fields)['category_contrib'].transform(np.sum)
        return df_corp_quotes

    @staticmethod
    def compute_weighted_average(dataframe, column_label_values, column_label_weights):
        """
        :param dataframe: dataframe containing values to compute weighted average
        :param column_label_values: - column with values to weight
        :param column_label_weights:
        :return: float - weighted average
        """

        weighted_average = (dataframe[column_label_values]*dataframe[column_label_weights]).sum()
        weighted_average /= dataframe[column_label_weights].sum()
        return weighted_average

    def serialize_curves_data(self):
        curves_quotes_data = []
        curves_stats_data = []
        curve_ids = set()
        curve_node_ids = set()
        trade_dates = set()
        cursor = self.curve_queries.macDBConn.cursor()
        tenor_enum = {k:v for k,v in self.curve_queries.fetch_mac_curves_query_result('select TenorEnum, InYears from [MarketData].[dbo].[TenorEnum]')}
        lub = 'DataGenOpsFl_Content'
        for curve_short_name, curve_data in CurveGenerator.curves_cache.iteritems():#keyed by curveshortname
            isABS = CurveGenerator.curves_metadata.get(curve_short_name,{}).get('isABS', False)
            isCMF = CurveGenerator.curves_metadata.get(curve_short_name,{}).get('isCMF', False)
            isCDX = CurveGenerator.curves_metadata.get(curve_short_name,{}).get('isCDX', False)
            curve_id = CurveGenerator.curves_metadata.get(curve_short_name,{}).get('curve_id')
            trade_dates = trade_dates.union(set(curve_data.keys()))
            query = 'select cn.TenorEnum,cn.CurveNodeId from [MarketData].[dbo].[CurveNodes] as cn where cn.CurveId=%d'%(curve_id)
            curve_nodes = {tenor_enum[tenor]  if not isCDX else float(tenor) :curve_node for tenor, curve_node in self.curve_queries.fetch_mac_curves_query_result(query)}
            if isCMF:
                self.serialize_cmf_curve_input_data(curve_short_name)
            for date in trade_dates:
                curve_stats = curve_data.get(date)
                if curve_stats is not None: ############
                    if 'corporate_curve_quotes' in curve_stats:
                        curve_ids.add(curve_id)
                        for curve_corporate_stats in curve_stats['corporate_curve_quotes']:
                            if isABS:
                                curves_quotes_data.append((self.derived_curve_filtered_bond_table_abs, str(curve_id), date, str(curve_corporate_stats['Isin']), 2, curve_corporate_stats['MatCorpYld'], lub))
                            else:
                                curves_quotes_data.append((self.derived_curve_filtered_bond_table, str(curve_id), date, str(curve_corporate_stats['InstrCode']), 2, curve_corporate_stats['MatCorpYld'], lub))
                if isinstance(curve_stats, dict):
                    for tenor, curve_node in curve_nodes.iteritems():
                        curve_node_ids.add(curve_node)
                        if 'amount_weighted_corporate_spread' in curve_stats:
                            curves_stats_data.append((str(curve_node), date, curve_stats['amount_weighted_corporate_spread'], lub))
                        elif 'sovereign_spread' in curve_stats:
                            curves_stats_data.append((str(curve_node), date, curve_stats['sovereign_spread'], lub))
                        elif 'covered_spread' in curve_stats:
                            if tenor in curve_stats['covered_spread']:
                                curves_stats_data.append((str(curve_node), date, curve_stats['covered_spread'][tenor], lub))
                elif isinstance(curve_stats, list):
                    for stat in curve_stats:
                        if stat['tenor'] in curve_nodes:
                            curve_node_ids.add(curve_nodes[stat['tenor']])
                            curves_stats_data.append((str(curve_nodes[stat['tenor']]), date, stat['quotes'],lub))
            logging.info('Populating records for %s'%(curve_short_name))
        if len(curve_ids):
            for chunk in Utilities.grouper(list(curve_ids), 1000):
                cursor.execute('delete from [MarketData].[dbo].[%s] where CurveId in (%s) and TradeDate in (%s)'%(self.derived_curve_filtered_bond_table
                                                                                                              ,','.join([str(curve_id) for curve_id in chunk])
                                                                                                              ,','.join(["'"+trade_date+"'" for trade_date in trade_dates])))
                cursor.execute('delete from [MarketData].[dbo].[%s] where CurveId in (%s) and TradeDate in (%s)'%(self.derived_curve_filtered_bond_table_abs
                                                                                                              ,','.join([str(curve_id) for curve_id in chunk])
                                                                                                              ,','.join(["'"+trade_date+"'" for trade_date in trade_dates])))                
                self.curve_queries.macDBConn.commit()
        if len(curve_node_ids):
            for chunk in Utilities.grouper(list(curve_node_ids), 1000):
                cursor.execute('delete from [MarketData].[dbo].[%s] where CurveNodeId in (%s) and TradeDate in (%s)'%(self.curve_node_quote_table
                                                                                                            ,','.join([str(curve_id) for curve_id in chunk])
                                                                                                            ,','.join(["'"+trade_date+"'" for trade_date in trade_dates])
                                                                                                            ))
                self.curve_queries.macDBConn.commit()
        if len(curves_quotes_data):
            abs_data = [data[1:] for data in curves_quotes_data if 'ABS' in data[0]]
            query = 'insert into [MarketData].[dbo].['+self.derived_curve_filtered_bond_table_abs+'] values (%s,%s,%s,%d,%s, GETDATE(),%s)'
            for chunk in Utilities.grouper(abs_data, 1000):
                cursor.executemany(query, chunk)
                logging.info('Updated %d rows for %s'%(cursor.rowcount,self.derived_curve_filtered_bond_table_abs))
                
            non_abs_data = [data[1:] for data in curves_quotes_data if 'ABS' not in data[0]]
            query = 'insert into [MarketData].[dbo].['+self.derived_curve_filtered_bond_table+'] values (%s,%s,%s,%d,%s, GETDATE(),%s)'
            for chunk in Utilities.grouper(non_abs_data, 1000):
                cursor.executemany(query, chunk)
                logging.info('Updated %d rows for %s'%(cursor.rowcount,self.derived_curve_filtered_bond_table))

        if len(curves_stats_data):
            query = 'insert into [MarketData].[dbo].['+self.curve_node_quote_table+'] values(%s, %s, %s, NULL, GETDATE(), %s)'
            for chunk in Utilities.grouper(curves_stats_data, 1000):
                try:
                    cursor.executemany(query, chunk)
                    logging.info('Updated %d rows for %s'%(cursor.rowcount,self.curve_node_quote_table))
                except Exception as e:
                    raise Exception(str(e) + '\nFailed Chunk:\n' + str(chunk))
        self.curve_queries.macDBConn.commit()
        logging.info('Committing changes')

    def serialize_cmf_curve_input_data(self, curve_short_name):
        cursor = self.curve_queries.macDBConn.cursor()
        curve_input_contracts = CurveGenerator.curves_input_contracts.get(curve_short_name,{})
        curve_id = CurveGenerator.curves_metadata.get(curve_short_name,{}).get('curve_id')
        lub = 'DataGenOpsFl_Content'
        if len(curve_input_contracts):
            for date, input_contracts in curve_input_contracts.iteritems():
                cursor.execute('delete from [MarketData].[dbo].[%s] where CurveId in (%s) and TradeDate in (%s)'%(self.derived_curve_filtered_bond_table_cmf, curve_id,  "'"+date+"'"))
                self.curve_queries.macDBConn.commit()
                for input_contract in input_contracts:
                    query = "insert into [MarketData].[dbo].[%s] values (%s,%f,%f,%s,%d,%f,%s,GETDATE(),%s)"%(self.derived_curve_filtered_bond_table_cmf, curve_id,
                                                                                                              input_contract['settlement_price'], input_contract['quotes'],
                                                                                                              "'"+input_contract['last_trade_date'].strftime('%Y-%m-%d')+"'", input_contract['future_contract_code'],
                                                                                                              input_contract['tenor'], "'"+date+"'", "'"+lub+"'")
                    cursor.execute(query)
                    self.curve_queries.macDBConn.commit()
    
    def serialize_curve_data(self, curve_short_name):
        curve_quotes_data = []
        curve_stats_data = []
        curve_node_ids = set()
        trade_dates = set()
        curve_id = CurveGenerator.curves_metadata.get(curve_short_name,{}).get('curve_id')
        isABS = CurveGenerator.curves_metadata.get(curve_short_name,{}).get('isABS', False)
        isCMF = CurveGenerator.curves_metadata.get(curve_short_name,{}).get('isCMF', False)
        isCDX = CurveGenerator.curves_metadata.get(curve_short_name,{}).get('isCDX', False)
        cursor = self.curve_queries.macDBConn.cursor()
        tenor_enum = {k:v for k,v in self.curve_queries.fetch_mac_curves_query_result('select TenorEnum, InYears from [MarketData].[dbo].[TenorEnum]')}
        lub = 'DataGenOpsFl_Content'
        curve_data = CurveGenerator.curves_cache.get(curve_short_name)
        if isCMF:
            self.serialize_cmf_curve_input_data(curve_short_name)
        if curve_data is not None:
            trade_dates = trade_dates.union(set(curve_data.keys()))
            query = 'select cn.TenorEnum,cn.CurveNodeId from [MarketData].[dbo].[CurveNodes] as cn where cn.CurveId=%d'%(curve_id)
            curve_nodes = {tenor_enum[tenor] if not isCDX else float(tenor) :curve_node for tenor, curve_node in self.curve_queries.fetch_mac_curves_query_result(query)}
            for date in trade_dates:
                curve_stats = curve_data.get(date)
                if curve_stats is not None:
                    if 'corporate_curve_quotes' in curve_stats:
                        for curve_corporate_stats in curve_stats['corporate_curve_quotes']:
                            if isABS:
                                curve_quotes_data.append((self.derived_curve_filtered_bond_table_abs, str(curve_id), date, str(curve_corporate_stats['Isin']), 2, curve_corporate_stats['MatCorpYld'], lub))
                            else:
                                curve_quotes_data.append((self.derived_curve_filtered_bond_table, str(curve_id), date, str(curve_corporate_stats['InstrCode']), 2, curve_corporate_stats['MatCorpYld'], lub))
                if isinstance(curve_stats, dict):
                    for tenor, curve_node in curve_nodes.iteritems():
                        curve_node_ids.add(curve_node)
                        if 'amount_weighted_corporate_spread' in curve_stats:
                            curve_stats_data.append((str(curve_node), date, curve_stats['amount_weighted_corporate_spread'], lub))
                        elif 'sovereign_spread' in curve_stats:
                            curve_stats_data.append((str(curve_node), date, curve_stats['sovereign_spread'], lub))
                        elif 'covered_spread' in curve_stats:
                            if tenor in curve_stats['covered_spread']:
                                curve_stats_data.append((str(curve_node), date, curve_stats['covered_spread'][tenor], lub))
                elif isinstance(curve_stats, list):
                    for stat in curve_stats:
                        if stat['tenor'] in curve_nodes:
                            curve_node_ids.add(curve_nodes[stat['tenor']])
                            curve_stats_data.append((str(curve_nodes[stat['tenor']]), date, stat['quotes'],lub))
            logging.info('Populating records for %s'%(curve_short_name))

            sql_delete = 'delete from [MarketData].[dbo].[%s] where CurveId in (%s) and TradeDate in (%s)'%(self.derived_curve_filtered_bond_table
                                                                                                      ,','.join([str(curve_id)])
                                                                                                      ,','.join(["'"+trade_date+"'" for trade_date in trade_dates]))

            cursor.execute(sql_delete)
            cursor.execute('delete from [MarketData].[dbo].[%s] where CurveId in (%s) and TradeDate in (%s)'%(self.derived_curve_filtered_bond_table_abs
                                                                                                      ,','.join([str(curve_id)])
                                                                                                      ,','.join(["'"+trade_date+"'" for trade_date in trade_dates])))     
            self.curve_queries.macDBConn.commit()
        
        else:
            logging.error('\nCurve %s was not serialized. No data found\n'%(curve_short_name))
            CurveGenerator.exit_status = 1

        if len(curve_node_ids):
            for chunk in Utilities.grouper(list(curve_node_ids), 1000):
                cursor.execute('delete from [MarketData].[dbo].[%s] where CurveNodeId in (%s) and TradeDate in (%s)'%(self.curve_node_quote_table
                                                                                                            ,','.join([str(curve_id) for curve_id in chunk])
                                                                                                            ,','.join(["'"+trade_date+"'" for trade_date in trade_dates])
                                                                                                            ))
                self.curve_queries.macDBConn.commit()
        if len(curve_quotes_data):
            abs_data = [data[1:] for data in curve_quotes_data if 'ABS' in data[0]]
            query = 'insert into [MarketData].[dbo].['+self.derived_curve_filtered_bond_table_abs+'] values (%s,%s,%s,%d,%s, GETDATE(),%s)'

            for chunk in Utilities.grouper(abs_data, 1000):
                #cursor.executemany(query, chunk)   ###################################
                logging.info('Updated %d rows for %s'%(cursor.rowcount,self.derived_curve_filtered_bond_table_abs))
                
            non_abs_data = [data[1:] for data in curve_quotes_data if 'ABS' not in data[0]]
            query = 'insert into [MarketData].[dbo].['+self.derived_curve_filtered_bond_table+'] values (%s,%s,%s,%d,%s, GETDATE(),%s)'
            for chunk in Utilities.grouper(non_abs_data, 1000):
                #cursor.executemany(query, chunk)  ########################################
                logging.info('Updated %d rows for %s'%(cursor.rowcount,self.derived_curve_filtered_bond_table))
        
        if len(curve_stats_data):
            query = 'insert into [MarketData].[dbo].['+self.curve_node_quote_table+'] values(%s, %s, %s, NULL, GETDATE(), %s)'
            for chunk in Utilities.grouper(curve_stats_data, 1000):
                try:
                    cursor.executemany(query, chunk)
                    logging.info('Updated %d rows for %s'%(cursor.rowcount,self.curve_node_quote_table))
                except Exception as e:
                    raise Exception(str(e) + '\nFailed Chunk:\n' + str(chunk))

        self.curve_queries.macDBConn.commit()
        logging.info('Committing changes')


def replace_null(val):
    if val is not None:
        return val
    return 0.0

def curve_runner(curve_generator, curve_short_name, start_date, end_date, window_size, num_windows, generate_history):
    curve_short_name = curve_short_name.strip("'")
    day1 = start_date
    day2 = day1 + datetime.timedelta(days=min(window_size, (end_date-start_date).days))
    curve_data = {}
    for window in xrange(0, num_windows):
        for date in curve_generator.calender_days_between(day1, day2):
            try:
                start_date_window = date.strftime('%Y-%m-%d')
                end_date_window = date.strftime('%Y-%m-%d')
                result = curve_generator.run_curve(curve_short_name, start_date_window, end_date_window)
                curve_data.update(result)
            except Exception as e:
                if CurveGenerator.curves_metadata.get(curve_short_name,{}).get('isactive', True):
                    err_msg = '{curve_name} generated an exception for {date}'.format(curve_name=curve_short_name, date=start_date_window)
                    logging.error(err_msg)
                    logging.error(traceback.format_exc())
                    CurveGenerator.exit_status = 1
            else:
                logging.info('Finished %s to %s for %s' % (day1.strftime('%Y-%m-%d'),
                                                       day2.strftime('%Y-%m-%d'), curve_short_name))
                day1 = day2 + datetime.timedelta(days=1)
            day2 = day1 + datetime.timedelta(days=window_size)

    for date_curve in curve_data:
        if 'amount_weighted_corporate_spread' in curve_data[date_curve]:
            logging.info(curve_short_name +' '+ str(date_curve) +' '+ str(curve_data[date_curve]['amount_weighted_corporate_spread']))
        elif 'sovereign_spread' in curve_data[date_curve]:
            logging.info(curve_short_name +' '+ str(date_curve) +' '+ str(curve_data[date_curve]['sovereign_spread']))

    if curve_generator.test_only is False:
        if ((curve_generator.run_curves_by_currency is False) and (curve_generator.run_curves_by_country is False) and (curve_generator.run_curves_by_type is False)) or generate_history == True:
            curve_generator.serialize_curve_data(curve_short_name)
    return curve_data


def run_currency_curves(currency_code, curve_generator, start_date, end_date, window_size, nWindows, generate_history):
    curve_query = curve_generator.curve_queries.curve_detail_by_currency(currency_code,
                                                                         start_date.strftime('%d-%b-%Y'),
                                                                         end_date.strftime('%d-%b-%Y'))

    curve_result = curve_generator.curve_queries.fetch_mac_curves_query_result(curve_query)

    curve_dict = [{'country_code': data[0], 'currency': data[1], 'curve_family': data[2],
                   'composite_rating': data[3], 'curve_short_name': data[4], 'curve_id': data[5], 'gics': data[6],
                   'underlying_curve_id': data[7], 'issuer_id': data[8]} for data in curve_result]

    curves = sorted(curve_dict, lambda x, y: sort_curves(x,y), reverse=True)

    for curve in curves:
        try:
            curve_data = curve_runner(curve_generator, curve['curve_short_name'], start_date, end_date, window_size, nWindows, generate_history)
            for date in curve_data:
                if 'amount_weighted_corporate_spread' in curve_data[date]:
                    logging.info(curve['curve_short_name'] +
                                 ' ' + str(date) + ' ' + str(curve_data[date]['amount_weighted_corporate_spread']))
                elif 'sovereign_spread' in curve_data[date]:
                    logging.info(curve['curve_short_name'] +
                                 ' ' + str(date) + ' ' + str(curve_data[date]['sovereign_spread']))
        except ValueError:
            logging.info('Curve %s is not active'%curve['curve_short_name'])
        except Exception as e:
            logging.info(traceback.format_exc())
            CurveGenerator.exit_status = 1


def run_curves_by_type(curve_types, curve_generator, start_date, end_date, window_size, num_windows, generate_history):
    curves = sorted([{'country_code': data[0], 'currency':data[1], 'curve_family':data[2], 'composite_rating': data[3],
                      'curve_short_name': data[4], 'curve_id': data[5], 'gics': data[6], 'underlying_curve_id': data[7],
                      'issuer_id': data[8]} for data in curve_generator.curve_queries.fetch_mac_curves_query_result(
        curve_generator.curve_queries.curve_detail_by_curvetypes(curve_types, start_date.strftime('%d-%b-%Y'),
                                                                         end_date.strftime('%d-%b-%Y')))],
                    lambda x, y: sort_curves(x, y), reverse=True)

    for curve in curves:
        try:
            curve_data = curve_runner(curve_generator, curve['curve_short_name'],start_date, end_date, window_size,
                                      num_windows, generate_history)
            for date in curve_data:
                if 'amount_weighted_corporate_spread' in curve_data[date]:
                    logging.info(curve['curve_short_name'] +
                                 ' ' + str(date) + ' ' + str(curve_data[date]['amount_weighted_corporate_spread']))
                elif 'sovereign_spread' in curve_data[date]:
                    logging.info(curve['curve_short_name'] +
                                 ' ' + str(date) + ' ' + str(curve_data[date]['sovereign_spread']))
        except ValueError:
            logging.info('Curve %s is not active'%curve['curve_short_name'])
        except Exception as e:
            logging.info(traceback.format_exc())
            CurveGenerator.exit_status = 1


def run_curves_by_type_currency(curve_types, currency_code, curve_generator, start_date, end_date, window_size,
                                num_windows, generate_history):
    curves = sorted([{'country_code': data[0], 'currency':data[1], 'curve_family': data[2], 'composite_rating': data[3],
                      'curve_short_name': data[4], 'curve_id': data[5], 'gics': data[6], 'underlying_curve_id': data[7],
                      'issuer_id': data[8]}
                     for data in curve_generator.curve_queries.fetch_mac_curves_query_result(
            curve_generator.curve_queries.curve_detail_by_curvetypes_currency(curve_types, currency_code, start_date.strftime('%d-%b-%Y'),
                                                                         end_date.strftime('%d-%b-%Y')))],
                    lambda x, y: sort_curves(x, y), reverse=True)

    for curve in curves:
        try:
            curve_data = curve_runner(curve_generator, curve['curve_short_name'],start_date, end_date, window_size,
                                      num_windows, generate_history)
            for date in curve_data:
                if 'amount_weighted_corporate_spread' in curve_data[date]:
                    logging.info(curve['curve_short_name'] +
                                 ' ' + str(date) + ' ' + str(curve_data[date]['amount_weighted_corporate_spread']))
                elif 'sovereign_spread' in curve_data[date]:
                    logging.info(curve['curve_short_name'] +
                                 ' ' + str(date) + ' ' + str(curve_data[date]['sovereign_spread']))
        except ValueError:
            logging.info('Curve %s is not active' % curve['curve_short_name'])
        except Exception as e:
            logging.info(traceback.format_exc())
            CurveGenerator.exit_status = 1

def run_curves_by_type_country(curve_types, country_code, curve_generator, start_date, end_date, window_size,
                                num_windows, generate_history):
    curves = sorted([{'country_code': data[0], 'currency':data[1], 'curve_family': data[2], 'composite_rating': data[3],
                      'curve_short_name': data[4], 'curve_id': data[5], 'gics': data[6], 'underlying_curve_id': data[7],
                      'issuer_id': data[8]}
                     for data in curve_generator.curve_queries.fetch_mac_curves_query_result(
            curve_generator.curve_queries.curve_detail_by_curvetypes_country(curve_types, country_code, start_date.strftime('%d-%b-%Y'),
                                                                         end_date.strftime('%d-%b-%Y')))],
                    lambda x, y: sort_curves(x, y), reverse=True)

    for curve in curves:
        try:
            curve_data = curve_runner(curve_generator, curve['curve_short_name'],start_date, end_date, window_size,
                                      num_windows, generate_history)
            for date in curve_data:
                if 'amount_weighted_corporate_spread' in curve_data[date]:
                    logging.info(curve['curve_short_name'] +
                                 ' ' + str(date) + ' ' + str(curve_data[date]['amount_weighted_corporate_spread']))
                elif 'sovereign_spread' in curve_data[date]:
                    logging.info(curve['curve_short_name'] +
                                 ' ' + str(date) + ' ' + str(curve_data[date]['sovereign_spread']))
        except ValueError:
            logging.info('Curve %s is not active' % curve['curve_short_name'])
        except Exception as e:
            logging.info(traceback.format_exc())
            CurveGenerator.exit_status = 1

def run_curves_by_country(country_code, curve_generator, start_date, end_date, window_size, nWindows, generate_history):
    curves = sorted([{'country_code':data[0], 'currency':data[1], 'curve_family':data[2], 'composite_rating':data[3],
                      'curve_short_name':data[4], 'curve_id':data[5], 'gics':data[6], 'underlying_curve_id':data[7],
                      'issuer_id':data[8]} for data in curve_generator.curve_queries.fetch_mac_curves_query_result(
        curve_generator.curve_queries.curve_detail_by_country(
            country_code, start_date.strftime('%d-%b-%Y'), end_date.strftime('%d-%b-%Y')))],
                    lambda x, y: sort_curves(x,y), reverse=True)
    for curve in curves:
        try:
            curve_data = curve_runner(curve_generator, curve['curve_short_name'], start_date, end_date, window_size, nWindows, generate_history)
            for date in curve_data:
                if 'amount_weighted_corporate_spread' in curve_data[date]:
                    logging.info(curve['curve_short_name'] +
                                 ' ' + str(date) + ' ' + str(curve_data[date]['amount_weighted_corporate_spread']))
                elif 'sovereign_spread' in curve_data[date]:
                    logging.info(curve['curve_short_name'] + ' ' + str(date) + ' ' +
                                 str(curve_data[date]['sovereign_spread']))
        except ValueError:
            logging.info('Curve %s is not active' % curve['curve_short_name'])
        except Exception as e:
            logging.info(traceback.format_exc())
            CurveGenerator.exit_status = 1


def unit_test(curve_short_name, curve_queries_instance, start_date, end_date):
    curve_id = curve_queries_instance.fetch_mac_curves_query_result(
        curve_queries_instance.curve_detail_by_shortname(curve_short_name))[0][5]

    def query(table):
        return r"""select cqr.CurveNodeID, cq.TenorEnum, cqr.Quote, cqr.TradeDate
          from [MarketData].[dbo].%s cqr, [MarketData].[dbo].[CurveNodes] cq
          where cqr.TradeDate >= '%s'
          and cqr.TradeDate <= '%s'
          and cqr.CurveNodeID = cq.CurveNodeID
          and cq.CurveID = %d""" % (table, start_date, end_date, curve_id)

    query_prod_result = curve_queries_instance.fetch_mac_curves_query_result(query("CurveNodeQuote"))
    query_research_result = curve_queries_instance.fetch_mac_curves_query_result(query("CurveNodeQuote_Research"))
    result = []
    if len(query_prod_result) and len(query_research_result):
        production_result = pd.DataFrame([{'curve_node_id': curve_node_id, 'trade_date': trade_date,
                                           'quote': quote, 'tenor': tenor}
                                          for curve_node_id, tenor, quote, trade_date in query_prod_result])
        research_result = pd.DataFrame([{'curve_node_id': curve_node_id, 'trade_date': trade_date, 'quote': quote,
                                         'tenor': tenor}
                                        for curve_node_id, tenor, quote, trade_date in query_research_result])
        df = production_result.merge(research_result, how='inner', on=['curve_node_id', 'trade_date'],
                                     suffixes=('_prod', '_research'))
        df['curve_short_name'] = curve_short_name
        df['parent_curve_short_name'] = CurveGenerator.curves_metadata.get(curve_short_name, {}).get(
            'parent_curve_short_name', None)
        df['absolute_deviation'] = (abs(df['quote_prod'] - df['quote_research'])/df['quote_prod'].abs())*100.0
        result.extend(df.T.to_dict().values())
    else:
        logging.info('Unit test not generated for %s' % curve_short_name)
    return result

def runner(mktDBInfo, modelDBInfo, macDBInfo, start_date, end_date, curve_short_name=None, currency_code=None,
           country_code=None, window_size=30, test_only=False, use_research_table=False, delete_existing_quotes=False,
           calculate_composite_rating=False, run_cmf=False, run_new_cmf = False, run_SvSprEmg = False,curve_types=None, generate_history = False, filename=None):

    num_windows = int(max(1, np.ceil((date_parser.parse(end_date)-date_parser.parse(start_date)).days/float(window_size))))
    curve_generator = CurveGenerator(mktDBInfo, modelDBInfo, macDBInfo,
                                     use_research = use_research_table,
                                     test_only = test_only,
                                     delete_existing_quotes=delete_existing_quotes,
                                     calculate_composite_rating=calculate_composite_rating,
                                     run_cmf=run_cmf, run_SvSprEmg = run_SvSprEmg,country_code = country_code, currency_code = currency_code, curve_types = curve_types)

    start_date = date_parser.parse(start_date)
    end_date = date_parser.parse(end_date)
    if (curve_types is not None) and (currency_code is not None):
        run_curves_by_type_currency(curve_types, currency_code, curve_generator, start_date, end_date, window_size, num_windows, generate_history)
    elif curve_short_name is not None:
        curve_runner(curve_generator, curve_short_name.strip(), start_date, end_date, window_size, num_windows, generate_history)
    elif currency_code is not None:
        run_currency_curves(currency_code, curve_generator, start_date, end_date, window_size, num_windows, generate_history)
    elif country_code is not None:
        query = curve_generator.curve_queries.country_code_to_currency(country_code, start_date.strftime('%d-%b-%Y'), end_date.strftime('%d-%b-%Y'))
        #query = curve_generator.curve_queries.country_code_to_currency_mssql(country_code)
        oracle_connection = Utilities.createOracleConnection(curve_generator.curve_queries.modelDBInfo)

        cursor = oracle_connection.cursor()
        #mssqlconnection = curve_generator.curve_queries.macDBConn
        #cursor = mssqlconnection.cursor()
        cursor.execute(query)
        currency_codes = list(cursor.fetchall())
        if len(currency_codes):
            for _currency_code in currency_codes:
                if curve_types is not None:
                     run_curves_by_type_currency(curve_types, _currency_code[0], curve_generator, start_date, end_date, window_size, num_windows, generate_history)
                else:
                    run_currency_curves(_currency_code[0], curve_generator, start_date, end_date, window_size, num_windows, generate_history)
    elif run_new_cmf is True:
        cmf_new_curves = ['CBOT:ZH', 'CBOT:ZZ', 'EEE:F1BY', 'KLSE:1FCPO', 'ICE:NGLNM', 'ICE:LRC', 'ICE:WTCL', 'MGE:1MWE', 'NYBOT:OJ', 'NYMEX:QG',
                      'ENP:BL2', 'ENP:COM', 'SAFE:WEA', 'SAFE:MAW', 'TCE:JAM', 'TCE:JAU', 'TCE:JCO', 'TCE:JGL', 'TCE:JKE', 'TCE:JPL', 'TCE:JRU',
                      'ICE:CERE', 'ICE CA:RS', 'CME:DCS', 'CBOT:RR', 'NYMEX:TIO', 'NYMEX:AAJU', 'CBOT:1DJE', 'NYMEX:BZZ', 'LME:MSN',
                      'ICE:CFI2',]
        for curve_short_name in cmf_new_curves:
            curve_runner(curve_generator, curve_short_name, start_date, end_date, window_size, num_windows, generate_history)
    elif run_cmf is True:
        run_curves_by_type(['Futures.CM'], curve_generator, start_date, end_date, window_size, num_windows, generate_history)
    elif run_SvSprEmg is True:
        run_curves_by_type(['SvSprEmg'], curve_generator, start_date, end_date, window_size, num_windows, generate_history)
    elif curve_types is not None:
        run_curves_by_type(curve_types, curve_generator, start_date, end_date, window_size, num_windows, generate_history)
    if test_only is False:
        if not generate_history:
            if (curve_generator.run_curves_by_currency is True) or (curve_generator.run_curves_by_country is True) or (curve_generator.run_curves_by_type is True):
                curve_generator.serialize_curves_data()
        result = []
        curves = sorted([{'country_code':data[0], 'currency':data[1], 'curve_family':data[2],
                          'composite_rating': data[3], 'curve_short_name':data[4], 'curve_id':data[5],
                          'gics':data[6], 'underlying_curve_id':data[7], 'issuer_id':data[8]}
                         for curve_short_name in CurveGenerator.curves_cache.keys()
                         for data in curve_generator.curve_queries.fetch_mac_curves_query_result(
                curve_generator.curve_queries.curve_detail_by_shortname(curve_short_name))],
                        lambda x, y: sort_curves(x, y), reverse=True)
        for curve in curves:
            result.extend(unit_test(curve['curve_short_name'], curve_generator.curve_queries,
                                    start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))

        if filename is not None:
            dirname=os.path.dirname(filename)
            if dirname <> '' and not os.path.isdir(dirname):
                os.makedirs(dirname)
            pd.DataFrame(result).reset_index().to_csv(filename)
    sys.exit(CurveGenerator.exit_status)

def sort_curves(curve2, curve1):
    curve2_family = curve2['curve_family']
    curve1_family = curve1['curve_family']
    
    if curve2_family == 'Sov.Zero':
        return 1
    elif curve1_family == 'Sov.Zero':
        return -1
    elif curve2_family == 'SwapZC':
        return 1
    elif curve1_family == 'SwapZC':
        return -1
    elif curve2_family == 'SwapZrSpr':
        return 1
    elif curve1_family == 'SwapZrSpr':
        return -1
    elif curve2_family == 'RtgSprSv':
        return 1
    elif curve1_family == 'RtgSprSv':
        return -1
    elif curve2_family == 'GiSprRat':
        return 1
    elif curve1_family == 'GiSprRat':
        return -1
    return 0

def arglist(astring):
    alist=astring.split(',')
    alist = [a.strip() for a in alist if len(a)]
    return alist if len(alist) else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("analysisConfig",
                        help="Input configuration file containing database connection info.",
                        action="store")
    parser.add_argument('-cmf',
                        dest="cmf_curve",
                        help="Run CMF curves",
                        action="store_true",
                        default=False)
    parser.add_argument('-cmf_new',
                        dest="cmf_new_curve",
                        help="Run new CMF curves",
                        action="store_true",
                        default=False)
    parser.add_argument('-SvSprEmg',
                        dest="SvSprEmg_curve",
                        help="Run Sovereign spread emerging market curves",
                        action="store_true",
                        default=False)
    parser.add_argument('-d', '--delete_existing_quotes',
                        help="Execute delete on all quotes within date range curves",
                        action="store_true",
                        default=False)
    parser.add_argument('-l', '--logConfigFile',
                        help="Optional full path to logging configuration file.",
                        action="store")
    parser.add_argument('-c', '--curve_short_name',
                        help="Name of the curve",
                        action="store")
    parser.add_argument('-e', '--environment',
                        help="Environment to use. Available options: PROD, UAT, DEVTEST. Default is DEV",
                        action="store",
                        default='DEV')
    parser.add_argument('-r', '--research',
                        help="Write to research CurveNodeQuote_Research table",
                        action="store_true",
                        default=False)
    parser.add_argument('-n',
                        help="Do not write results to database.",
                        dest="test_only",
                        action="store_true",
                        default=False)
    parser.add_argument("--currency",
                        help="Curve currency",
                        action="store",
                        default=None)
    parser.add_argument("--country_code",
                        help="Curve country code",
                        action="store",
                        default=None)
    parser.add_argument("--curve_types",
                        help="Curve types",
                        action="store",
                        type=arglist,
                        default=None)
    parser.add_argument("-m", "--calculate_composite_rating",
                        help="calculate the composite rating for the curve",
                        action="store_true",
                        default=False)
    parser.add_argument("--filename",
                        action="store",
                        default=None)
    parser.add_argument("-hist", "--generate_history",
                        help="Run in history mode or not",
                        action="store_true",
                        default=False)
    parser.add_argument('dates', action="store",
                        help="Date range to execute transfer. From-Thru: yyyy-mm-dd[:yyyy-mm-dd]",
                        metavar="DATE")
    args = parser.parse_args()

    #  --- Environment Info ---#
    environment = args.environment.replace('"', '').replace('\'', '').strip().upper()
    pwd = os.path.dirname(os.path.realpath(__file__)) + os.sep
    
    logConfigFile = args.logConfigFile if args.logConfigFile else os.path.join(pwd, "log.config")
    if not os.path.exists(logConfigFile):
        raise Exception("Logging configuration file:%s does not exist." % logConfigFile)
    logging.config.fileConfig(logConfigFile)
    
    configFile = open(pwd+"/production.config",'r')
    configuration = Utilities.loadConfigFile(configFile)

    sectionID = getAnalysisConfigSection(environment)
    envInfoMap = Utilities.getConfigSectionAsMap(configuration, sectionID)
    mktDBInfo = getOracleDatabaseInfo(Utilities.getConfigSectionAsMap(
        configuration, envInfoMap.get('equitymarketdb', None)))
    modelDBInfo = getOracleDatabaseInfo(Utilities.getConfigSectionAsMap(
        configuration, envInfoMap.get('equitymodeldb', None)))
    macDBInfo = getMSSQLDatabaseInfo(Utilities.getConfigSectionAsMap(
        configuration, envInfoMap.get('macdb', None)))

    dates = args.dates.split('=')[1] if args.dates and '=' in args.dates else None
    use_research_table = args.research
    delete_existing_quotes = args.delete_existing_quotes
    calculate_composite_rating = args.calculate_composite_rating

    start_date = None
    end_date = None
    if dates is not None:
        date_list = dates.split(':')
        if len(date_list) > 1:
            start_date = date_list[0]
            end_date = date_list[-1]
        else:
            start_date = date_list[0]
            end_date = start_date

    runner(mktDBInfo, modelDBInfo, macDBInfo, start_date, end_date, curve_short_name=args.curve_short_name,
           currency_code=args.currency, country_code=args.country_code, window_size=90, test_only=args.test_only,
           use_research_table=use_research_table, delete_existing_quotes=delete_existing_quotes,
           calculate_composite_rating=calculate_composite_rating, run_cmf=args.cmf_curve, run_new_cmf=args.cmf_new_curve, run_SvSprEmg = args.SvSprEmg_curve, curve_types=args.curve_types,
           generate_history=args.generate_history,filename=args.filename)

if __name__ == '__main__':
    try:
        __IPYTHON__
        print '\nrunning via ipython -> not running continously'
    except NameError:
        main()
