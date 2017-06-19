import string
import datetime
import macpy.utils.database as db
import pandas as pd
import numpy as np
from optparse import OptionParser
from dateflow import DateFlow
import traceback
from collections import Counter
import argparse
import logging
from dateutil import parser as date_parser
from itertools import groupby
import traceback
import os
import macpy.utils.log as log
from contextlib import contextmanager
from utils import Utilities
from macpy.CurveGenerator import CurveGenerator
from scipy import interp
from curve_utility import getAnalysisConfigSection, getOracleDatabaseInfo, getMSSQLDatabaseInfo


class CorpUniverseTableExtraction:
    """
        Pulls corp bonds from market data and inserts results into table
    """

    @classmethod
    def create_constructor(cls):
        def constructor(arg):
            return cls(arg)
        return constructor

    def __init__(self, extraction_args):
        """
        corp = CorpUniverseTableExtraction('2001-01-01','2001-01-01', mac_database)
        df = corp.extract_corp_universe()
        """
        self.mac_database = extraction_args.database
        self.start_date = extraction_args.start_date
        self.end_date = extraction_args.end_date
        self.currency = extraction_args.currency

        self.sql_ccy_template = """
        USE MarketData
        Declare @CurrencyCode        varchar(3)  =  '{0}';
        Declare @TradeDateBegin      datetime  =  '{1}';
        Declare @TradeDateEnd        datetime  =  '{2}';
        SELECT TradeDate, sec.InitAccDate, sec.MatDate, Country, Currency, cu.InstrCode, Isin, IsCallable, OrgName,
            UltParIsrId, Amt, Prc, sec.CurrCpnRate, WrstCorpYld, MatCorpYld,
            RatingCurveShortName = rating.CurveShortName,
            SectorCurveShortName = sector.CurveShortName,
            IssuerCurveShortName = issuer.CurveShortName,
            cu.CompositeRatingEnum,
            GicsLevel1,
            UltParIsrShortName
        FROM [MarketData].[dbo].[DerivCurveCorpBondUniverse] cu
        join QAI..FIEJVSecInfo sec on sec.InstrCode = cu.InstrCode and cu.Currency = @CurrencyCode
        left join Curve rating on rating.CurrencyEnum=cu.Currency
            and rating.CompositeRatingEnum=cu.CompositeRatingEnum
            and rating.CurveTypeEnum='RtgSprSv'
            and rating.RegionEnum is null
            and rating.CurveShortName not like '%Callable%'
            and rating.CurrencyEnum = @CurrencyCode
        left join Curve sector on sector.CurrencyEnum=cu.Currency
            and sector.CompositeRatingEnum=cu.CompositeRatingEnum
            and sector.GicsEnum=cu.GicsLevel1
            and sector.CurveTypeEnum='GiSprRat'
            and sector.RegionEnum is null
            and sector.CurveShortName not like '%Callable%'
            and sector.CurrencyEnum = @CurrencyCode
        left join Curve issuer on issuer.CurrencyEnum=cu.Currency
            and issuer.IssuerId = cu.UltParIsrId
            and issuer.CurveTypeEnum='IssrSpr'
            and issuer.RegionEnum is null
            and issuer.CurveShortName not like '%Callable%'
            and issuer.CurrencyEnum = @CurrencyCode
        WHERE TradeDate >= @TradeDateBegin and TradeDate <= @TradeDateEnd
        """
        self.sql_ccy = self.sql_ccy_template.format(self.currency, self.start_date, self.end_date)

    def extract_corp_universe(self, logger_callback=log.create_null_logger()):
        df_ccy = self.mac_database.extract_dataframe(self.sql_ccy, logger_callback=logger_callback)
        return df_ccy


class CorpUniverseExtraction:
    """
        Pulls corp bonds from market data and inserts results into table
    """

    @classmethod
    def create_constructor(cls):
        def constructor(arg):
            return cls(arg)
        return constructor

    def __init__(self, extraction_args):
        """
        corp = CorpUniverseExtraction('2001-01-01','2001-01-01', mac_database)
        df = corp.extract_corp_universe()
        """
        self.mac_database = extraction_args.database
        self.start_date = extraction_args.start_date
        self.end_date = extraction_args.end_date
        self.currency = extraction_args.currency

        self.sql_ccy_template = """
        exec [MarketData].[dbo].[DerivedDataCorpBondPriceForCcyWithRating]
            @tradeDateBegin = '{0}',
            @tradeDateEnd = '{1}',
            @currencyISO = '{2}',
            @IncCallables = 'Y'
        """
        self.sql_ccy = self.sql_ccy_template.format(self.start_date, self.end_date, self.currency)

    def extract_corp_universe(self, logger_callback=log.create_null_logger()):
        df_ccy = self.mac_database.extract_dataframe(self.sql_ccy, logger_callback=logger_callback)
        df_ccy.drop_duplicates(subset=['TradeDate', 'InstrCode'], inplace=True)
        df_ccy.dropna(subset=['WrstCorpYld', 'MatCorpYld', 'Amt'], inplace=True)  # drop empty values
        df_ccy.loc[df_ccy['UltParIsrId'].isnull(), 'UltParIsrId'] = 0  # set ultimate parent id to 0 if null
        df_ccy.loc[:, 'OrgName'] = df_ccy.loc[:, 'OrgName'].str.replace("'", '')

        return df_ccy


class ExtractionArgument:
    """
    Container class for arguments used in stored procedures
    """

    def __init__(self, start_date, end_date, currency, database):
        self.start_date = start_date
        self.end_date = end_date
        self.currency = currency
        self.database = database

    def iterate_business_days(self):
        for date in pd.bdate_range(start=self.start_date, end=self.end_date, freq='B'):
            yield ExtractionArgument(date, date, self.currency, self.database)


class JobRunner:

    def __init__(self, extraction, extraction_args, insertion, logger=log.create_info_logger()):
        self.create_extraction = extraction
        self.extraction_args = extraction_args
        self.create_insertion = insertion
        self.logger = logger
        self.total_extracted = 0
        self.total_inserted = 0

    def run(self):
        log_str = 'Running {} {}:{}'.format(self.extraction_args.currency, self.extraction_args.start_date,
                                            self.extraction_args.end_date)
        self.logger(log_str)
        business_days = [x for x in self.extraction_args.iterate_business_days()]
        for arg in business_days:
            log_str = 'Running Date Range {}:{}'.format(arg.start_date, arg.end_date)
            self.logger(log_str)
            extraction_procedure = self.create_extraction(arg)
            df_result = extraction_procedure.extract_corp_universe(logger_callback=self.logger)
            corp_insert = self.create_insertion(extraction_procedure, dataframe=df_result)
            expected_count = len(df_result)
            self.logger('Extracted {} rows from CCY'.format(expected_count))
            sql_delete = corp_insert.create_delete_sql()
            corp_insert.execute_delete(logger_callback=self.logger)
            sql_insert_list = corp_insert.create_insert_sql()
            corp_insert.execute_insert(logger_callback=self.logger)
            actual_count = corp_insert.execute_row_count()
            self.logger('Inserted {} rows'.format(actual_count))
            self.total_extracted += expected_count
            self.total_inserted += actual_count

        self.logger('Extracted total {} rows'.format(self.total_extracted))
        self.logger('Inserted total {} rows'.format(self.total_inserted))


class CorpUniverseInsertion:
    """
        Pulls corp bonds from market data and inserts results into table
    """

    @classmethod
    def create_constructor(cls):
        def constructor(corp_extraction, dataframe):
            return cls(corp_extraction, dataframe)
        return constructor

    def __init__(self, corp_extraction, dataframe=None, mac_database=None):
        self.corp_extraction = corp_extraction
        self.dataframe = dataframe if dataframe is not None else corp_extraction.extract_corp_universe()
        self.mac_database = corp_extraction.mac_database if mac_database is None else mac_database
        self.sql_insert_template = r"""
        INSERT INTO [dbo].[DerivCurveCorpBondUniverse]
                   ([TradeDate]
                   ,[MatDate]
                   ,[InstrCode]
                   ,[Isin]
                   ,[IsCallable]
                   ,[OrgName]
                   ,[CompositeRatingEnum]
                   ,[GicsLevel1]
                   ,[UltParIsrId]
                   ,[UltParIsrShortName]
                   ,[Amt]
                   ,[Prc]
                   ,[WrstCorpYld]
                   ,[MatCorpYld]
                   ,[Lud]
                   ,[Lub]
                   ,[Country]
                   ,[Currency])
             VALUES
                   ('{:%Y-%m-%d}' --<TradeDate, date,>
                   ,'{:%Y-%m-%d}' --<MatDate, date,>
                   ,{} --<InstrCode, int,>
                   ,'{}' --<Isin, varchar(12),>
                   ,{} --<IsCallable, int,>
                   ,'{}' --<OrgName, varchar(300),>
                   ,'{}' --<CompositeRatingEnum, varchar(10),>
                   ,'{}' --<GicsLevel1, varchar(300),>
                   ,{} --<UltParIsrId, int,>
                   ,'{}' --<UltParIsrShortName, varchar(300),>
                   ,{} --<Amt, int,>
                   ,{} --<Prc, float,>
                   ,{} -- <WrstCorpYld, float,>
                   ,{} -- <MatCorpYld, float,>
                   ,GETDATE()
                   ,'DataGenOpsFl_Content'
                   ,'{}'
                   ,'{}')
                """

        self.sql_count_template = r"""
        SELECT Count(InstrCode) as InstrCodeCount
        FROM DerivCurveCorpBondUniverse
        WHERE TradeDate >= '{}' and TradeDate <= '{}' and Currency='{}'
        """

        self.sql_delete_template = r"""
        DELETE DerivCurveCorpBondUniverse
        WHERE TradeDate >= '{}' and TradeDate <= '{}' and Currency='{}'
        """

    def create_insert_sql(self):
        df_results = self.dataframe
        df_results.loc[df_results['UltParIsrId'].isnull(), 'UltParIsrId'] = 0  # set ultimate parent id to 0 if null
        df_results.loc[:, 'OrgName'] = df_results.loc[:, 'OrgName'].str.replace("'", '')
        fields = ['TradeDate', 'MatDate', 'InstrCode', 'Isin', 'IsCallable', 'OrgName', 'CompositeRatingEnum',
                  'GicsLevel1', 'UltParIsrId', 'UltParIsrShortName', 'Amt', 'Prc', 'WrstCorpYld', 'MatCorpYld',
                  'ISOCtryCode', 'DebtISOCurCode']
        records = df_results[fields].values
        sql_insert_list = [self.sql_insert_template.format(*record) for record in records]
        return sql_insert_list

    def create_delete_sql(self):
        delete_sql = self.sql_delete_template.format(self.corp_extraction.start_date, self.corp_extraction.end_date,
                                                     self.corp_extraction.currency)
        return delete_sql

    def create_sql_count(self):
        row_count_sql = self.sql_count_template.format(self.corp_extraction.start_date, self.corp_extraction.end_date,
                                                       self.corp_extraction.currency)
        return row_count_sql

    def extract_corp_universe(self):
        return self.corp_extraction.extract_corp_universe()

    def execute_insert(self, logger_callback=log.create_null_logger()):
        sql_insert_list = self.create_insert_sql()
        self.mac_database.executemany(sql_insert_list, logger_callback)

    def execute_delete(self, logger_callback=None):
        sql_delete_list = self.create_delete_sql()
        self.mac_database.execute(sql_delete_list, logger_callback=logger_callback)

    def execute_row_count(self, logger_callback=None):
        row_count_sql = self.sql_count_template.format(self.corp_extraction.start_date, self.corp_extraction.end_date,
                                                       self.corp_extraction.currency)
        df_result = self.mac_database.extract_dataframe(row_count_sql)
        count = df_result.InstrCodeCount[0]
        return count


class CorpSpreadCalculator:
    """
    Processes dataframe with fields below, returns corporate spread by issuer, sector, rating for each curve short name
    TradeDate	1/1/2016
    MatDate	3/9/2023
    Amt	3000000000
    Prc	86.382
    WrstCorpYld	6.83024
    MatCorpYld	6.83024
    RatingCurveShortName	USD.(A).SPRSWP
    SectorCurveShortName	USD.RATGICsSEC(A,Financials)  -- optional
    IssuerCurveShortName	USD.ISSR(1MDB_GLOBAL_INVESTMENTS).SPR   -- optional
    """

    def __init__(self, dataframe=None, environment='DEV', logger=log.create_null_logger(), apply_filters=True,
                 clip_outliers=True):
        self.sector_curve_label = 'SectorCurveShortName'
        self.issuer_curve_label = 'IssuerCurveShortName'
        self.dataframe = dataframe
        self.environment = environment
        self.logger = logger
        self.clip_outliers = clip_outliers
        if apply_filters:
            self._apply_filters()
        self.df_corp = self.calculate()

    def _apply_filters(self):
        self.dataframe = self.dataframe[self.dataframe.Amt > 0]
        self.dataframe = self.dataframe[self.dataframe.Prc > 0.0]
        self.dataframe = self.dataframe[self.dataframe.WrstCorpYld > 0.0]
        self.dataframe = self.dataframe[self.dataframe.WrstCorpYld < 50.0]
        self.dataframe = self.dataframe[self.dataframe.WrstCorpYld >= 0.9 * self.dataframe.MatCorpYld]

    def calculate(self):
        """
        Requires database connection to retrieve swap yield curves
        Computes spread hierarchy - using the following fields in self.dataframe:
         TradeDate
         MatDate
         Prc
         MatCorpYld
         Amt
         RatingCurveShortName
         SectorCurveShortName -- optional
         IssuerCurveShortName -- optional
        """
        df_corp = pd.DataFrame(self.dataframe)  # create full copy of dataframe
        # pull swap spread curve
        country_code = df_corp.Currency[0][0:2]
        df_corp.MatDate = pd.to_datetime(df_corp.MatDate, infer_datetime_format=True)
        df_corp.TradeDate = pd.to_datetime(df_corp.TradeDate, infer_datetime_format=True)
        days_in_year_fraction = 365.25
        df_corp['TermInYears'] = \
            pd.to_timedelta(df_corp.MatDate -
                            df_corp.TradeDate).astype('timedelta64[D]').astype(int)/days_in_year_fraction
        df_corp['CorpYield'] = df_corp.MatCorpYld / 100.0
        df_corp.set_index('TradeDate', inplace=True, drop=False)
        curve_generator = CurveGenerator.create_curve_generator(environment=self.environment)
        for trade_date in df_corp.index.unique():
            trade_date_str = trade_date.strftime('%Y-%m-%d')
            df_swap_yield = curve_generator.get_swap_yield_quotes(
                country_code,
                trade_date_str,
                trade_date_str)

            if len(df_swap_yield.loc[trade_date_str]) == 0:
                self.logger("No swap quotes for trade date: " + trade_date_str)
                continue

            df_corp.loc[trade_date, 'SwapYieldInterp'] = interp(df_corp.loc[trade_date, 'TermInYears'],
                                                                df_swap_yield.loc[trade_date_str].index,
                                                                df_swap_yield.loc[trade_date_str, 'Quote'].values)

        # drop days where swap curve is missing - cannot compute swap spread for these days
        df_corp.dropna(inplace=True, subset=['SwapYieldInterp'])

        df_corp['CorpYieldSpreadOverSwap'] = df_corp.CorpYield - df_corp.SwapYieldInterp

        # compute pre-clip average and standard deviation
        df_avg = pd.DataFrame(df_corp.groupby(['TradeDate', 'RatingCurveShortName']).apply(
            lambda grp: (grp.CorpYieldSpreadOverSwap * grp.Amt).sum()/grp.Amt.sum())
            , columns=['RatGrpWAvgSpreadPreClip'])
        df_std = pd.DataFrame(df_corp.groupby(['TradeDate', 'RatingCurveShortName']).apply(
            lambda grp: (grp.CorpYieldSpreadOverSwap * grp.Amt).sum()/grp.Amt.sum())
            , columns=['RatGrpStdPreClip'])

        df_corp = df_corp.join(df_avg, on=['TradeDate', 'RatingCurveShortName'])  # broadcast group stats to all rows
        df_corp = df_corp.join(df_std, on=['TradeDate', 'RatingCurveShortName'])  # broadcast group stats to all rows

        # Clip outliers based on cross sectional standard deviation of spread over swap
        std_threshold = 3.0
        df_corp['SpreadClipUpperBound'] = df_corp.RatGrpStdPreClip + std_threshold * df_corp.RatGrpStdPreClip
        df_corp['SpreadClipLowerBound'] = df_corp.RatGrpStdPreClip - std_threshold * df_corp.RatGrpStdPreClip
        df_corp['Clipped'] = (df_corp.CorpYieldSpreadOverSwap > df_corp.SpreadClipUpperBound) | \
                             (df_corp.CorpYieldSpreadOverSwap < df_corp.SpreadClipLowerBound)

        if self.clip_outliers:
            df_corp = pd.DataFrame(df_corp[~df_corp.Clipped])

        # compute rating group yield spread over swap
        df_rat = pd.DataFrame(df_corp.groupby(['TradeDate', 'RatingCurveShortName']).apply(
            lambda grp: (grp.CorpYieldSpreadOverSwap * grp.Amt).sum()/grp.Amt.sum())
            , columns=['RatingGroupWAvgSpread'])
        df_corp = df_corp.join(df_rat, on=['TradeDate', 'RatingCurveShortName'])  # broadcast group average to all rows

        if self.sector_curve_label not in df_corp.columns:
            return df_corp

        # compute sector group yield spread over (rating spread + swap yield)
        df_sect = pd.DataFrame(df_corp.groupby(['TradeDate', 'SectorCurveShortName']).apply(
            lambda grp: ((grp.CorpYield - (grp.RatingGroupWAvgSpread + grp.SwapYieldInterp)) *
                         grp.Amt).sum()/grp.Amt.sum()),
            columns=['SectorGroupWAvgSpread'])

        df_corp = df_corp.join(df_sect, on=['TradeDate', 'SectorCurveShortName'])  # broadcast group average to all rows

        if self.issuer_curve_label not in df_corp.columns:
            return df_corp

        # compute issuer yield over (sector spread + rating spread + swap yield)
        df_issr_sector = pd.DataFrame(
            df_corp.groupby(['TradeDate', 'IssuerCurveShortName', 'SectorCurveShortName']).apply(
                lambda grp: ((grp.CorpYield -
                              (grp.RatingGroupWAvgSpread + grp.SectorGroupWAvgSpread + grp.SwapYieldInterp)) *
                             grp.Amt).sum()/grp.Amt.sum()),
            columns=['IssuerGroupWAvgSpreadBySector'])

        df_issr_sector_amt = pd.DataFrame(
            df_corp.groupby(['TradeDate', 'IssuerCurveShortName', 'SectorCurveShortName']).apply(
                lambda grp: grp.Amt.sum()),
            columns=['IssuerGroupWAvgAmtBySector'])

        df_corp = df_corp.join(df_issr_sector, on=['TradeDate', 'IssuerCurveShortName', 'SectorCurveShortName'])
        df_corp = df_corp.join(df_issr_sector_amt, on=['TradeDate', 'IssuerCurveShortName', 'SectorCurveShortName'])

        df_issr = pd.DataFrame(
            df_corp.groupby(['TradeDate', 'IssuerCurveShortName']).apply(
                lambda grp: (grp.IssuerGroupWAvgSpreadBySector *
                             grp.IssuerGroupWAvgAmtBySector).sum() / grp.IssuerGroupWAvgAmtBySector.sum()),
            columns=['IssuerGroupWAvgSpread'])

        df_corp = df_corp.join(df_issr, on=['TradeDate', 'IssuerCurveShortName'])  # broadcast group average to all rows

        return df_corp


class CommandLineGenerator:
    """
        Splits date range to monthly chunks
    """

    def __init__(self, start_date, end_date, command_line):
        self.start_date = start_date
        self.end_date = end_date
        self.command_line = command_line

    def create_date_list(self):
        end_date = date_parser.parse(self.end_date)
        date = date_parser.parse(self.start_date)
        date_list = []
        while date <= end_date:
            if date.month == 12:
                next_month = datetime.datetime(date.year + 1, 1, 1)
            else:
                next_month = datetime.datetime(date.year, date.month + 1, 1)
            eom = next_month - datetime.timedelta(days=1)
            filename = ""
            command_line = "{command_line} --dates={start:%Y-%m-%d}:{end:%Y-%m-%d}".format(
                command_line=self.command_line, start=date, end=eom)
            date = next_month
            date_list.append(command_line)

        return date_list


def create_config_file_path():
    pwd = os.path.dirname(os.path.realpath(__file__)) + os.sep
    log.activate_logging(working_directory=pwd)
    config_file_path = pwd+"/production.config"
    return config_file_path


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--currency',
                        dest="currency",
                        help="ISO Currency (e.g. USD, GBP) ",
                        action="store")
    parser.add_argument('-e', '--environment',
                        help="Environment to use. Available options: PROD, UAT, DEVTEST. Default is DEV",
                        action="store",
                        default='DEV')
    parser.add_argument('-d', '--dates',
                        action="store",
                        help="Date range to execute transfer. From-Thru: yyyy-mm-dd[:yyyy-mm-dd]",
                        metavar="DATE")

    args = parser.parse_args()
    config_file_path = create_config_file_path()
    start_date, end_date = Utilities.parse_dates(args.dates)
    environment = args.environment
    mac_database = db.MacDatabase(config_file_path, environment=environment)
    currency = args.currency
    argument = ExtractionArgument(start_date=start_date, end_date=end_date, currency=currency, database=mac_database)
    logger = log.create_info_logger()
    job = JobRunner(extraction=CorpUniverseExtraction.create_constructor(),
                    extraction_args=argument,
                    insertion=CorpUniverseInsertion.create_constructor(),
                    logger=logger)
    job.run()


if __name__ == '__main__':
        main()

