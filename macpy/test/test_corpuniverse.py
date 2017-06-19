import unittest
import macpy.corpuniverse as cu
import macpy.utils.database as db
import macpy.utils.log as log
import pandas as pd
import os


class Environment(object):

    pwd = os.path.dirname(os.path.realpath(__file__)) + """\.."""
    config_file_path = os.path.join(pwd, 'production.config')


class TestMacDatabase(unittest.TestCase):
    """
    Tests MacDatabase class
    """

    def test_main_use_dev(self):
        environment = 'DEV'
        sql_query = 'select top 1 * from Curve'
        mac_database = db.MacDatabase(Environment.config_file_path, environment)
        df_result = mac_database.extract_dataframe(sql_query)
        actual_rows = len(df_result)
        expected_rows = 1
        self.assertEquals(actual_rows, expected_rows)


class TestCorpUniverseExtraction(unittest.TestCase):

    def test_main_use_dev(self):
        environment = 'DEV'
        mac_database = db.MacDatabase(config_file_path, environment)
        start_date = '2006-11-29'
        end_date = '2006-11-29'
        currency = 'JPY'
        corp_creator = cu.CorpUniverseExtraction.create_constructor()
        argument = cu.ExtractionArgument(start_date, end_date, currency, mac_database)
        corp = corp_creator(argument)
        df_result = corp.extract_corp_universe()
        actual_rows = len(df_result)
        expected_rows = 7541
        self.assertEquals(actual_rows, expected_rows)

    def test_extraction_arguments(self):
        start_date = '2006-11-01'
        end_date = '2006-11-30'
        currency = 'JPY'
        argument = cu.ExtractionArgument(start_date, end_date, currency, None)
        business_days = [x.start_date for x in argument.iterate_business_days()]
        expected_number_days = 22
        actual_number_days = len(business_days)
        print business_days
        self.assertEqual(expected_number_days, actual_number_days)


class TestCorpUniverseInsertion(unittest.TestCase):

    def test_main_insert_sql(self):
        environment = 'DEV'
        mac_database = db.MacDatabase(Environment.config_file_path, environment)
        start_date = '2006-11-29'
        end_date = '2006-11-29'
        currency = 'JPY'
        test_instrcode = 450296
        corp = cu.CorpUniverseExtraction(cu.ExtractionArgument(start_date, end_date, currency, mac_database))
        df_result = corp.extract_corp_universe()

        corp_insert = cu.CorpUniverseInsertion(corp, dataframe=df_result)
        sql_insert_list = corp_insert.create_insert_sql()
        actual_sql = sql_insert_list[0]
        expected_sql = """
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
                   ('2006-11-29' --<TradeDate, date,>
                   ,'2017-08-12' --<MatDate, date,>
                   ,450296 --<InstrCode, int,>
                   ,'FR0000480337' --<Isin, varchar(12),>
                   ,0 --<IsCallable, int,>
                   ,'COMPAGNIE DE FINANCEMENT FONCIER SA' --<OrgName, varchar(300),>
                   ,'AAA' --<CompositeRatingEnum, varchar(10),>
                   ,'Financials' --<GicsLevel1, varchar(300),>
                   ,100004916.0 --<UltParIsrId, int,>
                   ,'BPCE_SA' --<UltParIsrShortName, varchar(300),>
                   ,500000000.0 --<Amt, int,>
                   ,112.372714 --<Prc, float,>
                   ,1.734658 -- <WrstCorpYld, float,>
                   ,1.734658 -- <MatCorpYld, float,>
                   ,GETDATE()
                   ,'DataGenOpsFl_Content'
                   ,'FR'
                   ,'JPY')
                """

        self.assertEqual(actual_sql, expected_sql)

    def test_main_count_sql(self):
        environment = 'DEV'
        mac_database = db.MacDatabase(Environment.config_file_path, environment)
        start_date = '2006-11-29'
        end_date = '2006-11-29'
        start_date = '2016-06-29'
        end_date = '2016-06-30'
        currency = 'JPY'
        corp_extract = cu.CorpUniverseExtraction(cu.ExtractionArgument(start_date, end_date, currency, mac_database))
        df_result = corp_extract.extract_corp_universe()
        corp_insert = cu.CorpUniverseInsertion(corp_extract, dataframe=df_result)
        expected_count = len(df_result)
        sql_delete = corp_insert.create_delete_sql()
        corp_insert.execute_delete()
        sql_insert_list = corp_insert.create_insert_sql()
        corp_insert.execute_insert(logger_callback=log.create_print_logger())
        actual_count = corp_insert.execute_row_count()
        self.assertEqual(expected_count, actual_count)

    def test_date_generator_group_year(self):
        environment = 'DEV'
        start_date = '2006-01-01'
        end_date = '2007-12-31'
        command_line = 'python CurveGenerator.py -c USD'
        cmd = cu.CommandLineGenerator(start_date, end_date, command_line)
        actual_date_list = cmd.create_date_list()
        expected_date_list = command_line + ' --dates={}:{}'.format(start_date, end_date)
        print actual_date_list
        self.assertEqual(expected_date_list, actual_date_list[0])

    def test_date_generator(self):
        environment = 'DEV'
        start_date = '2006-01-29'
        end_date = '2006-11-29'
        command_line = 'python CurveGenerator.py -c USD'
        cmd = cu.CommandLineGenerator(start_date, end_date, command_line)
        actual_date_list = cmd.create_date_list()
        expected_date_list = command_line + ' --dates={}:{}'.format(start_date, end_date)
        print actual_date_list
        self.assertEqual(expected_date_list, actual_date_list[0])

    def test_date_generator_large_range(self):
        environment = 'DEV'
        start_date = '2006-01-01'
        end_date = '2010-12-31'
        command_line = 'python CurveGenerator.py -c JPY'
        cmd = cu.CommandLineGenerator(start_date, end_date, command_line)
        actual_date_list = cmd.create_date_list()
        expected_date_list = command_line + ' --dates={}:{}'.format(start_date, end_date)
        self.assertEqual(actual_date_list[0], expected_date_list)

    def test_derived_job_runner(self):
        environment = 'DEV'
        start_date = '2006-01-03'
        end_date = '2006-01-03'
        currency = 'JPY'
        mac_database = db.MacDatabase(cu.create_config_file_path(), environment)
        argument = cu.ExtractionArgument(start_date=start_date, end_date=end_date, currency=currency,
                                         database=mac_database)
        job = cu.JobRunner(extraction=cu.CorpUniverseExtraction.create_constructor(),
                           extraction_args=argument,
                           insertion=cu.CorpUniverseInsertion.create_constructor(),
                           logger=log.create_null_logger())

        job.run()
        expected_total_inserted = 612
        actual_total_inserted = job.total_inserted
        self.assertEqual(expected_total_inserted, actual_total_inserted)

    def test_derived_job_runner_2008(self):
        environment = 'PROD'
        start_date = '2008-01-02'
        end_date = '2008-01-02'
        currency = 'USD'
        mac_database = db.MacDatabase(cu.create_config_file_path(), environment)
        argument = cu.ExtractionArgument(start_date=start_date, end_date=end_date, currency=currency,
                                         database=mac_database)
        job = cu.JobRunner(extraction=cu.CorpUniverseExtraction.create_constructor(),
                           extraction_args=argument,
                           insertion=cu.CorpUniverseInsertion.create_constructor(),
                           logger=log.create_null_logger())

        job.run()
        expected_total_inserted = 612
        actual_total_inserted = job.total_inserted
        self.assertEqual(expected_total_inserted, actual_total_inserted)

    def test_corp_curve_calculator(self):
        environment = 'DEV'
        csv_path = 'CorpBondUniverseTestData.csv'
        df_corp_quotes = pd.read_csv(csv_path)
        calculator = cu.CorpSpreadCalculator(df_corp_quotes, environment=environment)
        expected_spr_swp_usd = 0.01284
        actual_spr_swp_usd = calculator.df_corp[
            calculator.df_corp.RatingCurveShortName == 'USD.(A).SPRSWP']['RatingGroupWAvgSpread'].unique()[0]
        self.assertAlmostEqual(expected_spr_swp_usd, actual_spr_swp_usd, delta=.00005)

    def test_corp_curve_calculator_multiple_dates(self):
        environment = 'DEV'
        csv_path = 'CorpBondUniverseTestDataMultipleDates.csv'
        df_corp_quotes = pd.read_csv(csv_path)
        calculator = cu.CorpSpreadCalculator(df_corp_quotes, environment=environment)
        expected_spr_swp_usd = 0.0147188998748
        actual_spr_swp_usd = calculator.df_corp[
            calculator.df_corp.RatingCurveShortName == 'USD.(A).SPRSWP']['RatingGroupWAvgSpread'].unique()[0]
        self.assertAlmostEqual(expected_spr_swp_usd, actual_spr_swp_usd, delta=.00005)

    def test_corp_table_extraction(self):
        environment = 'DEV'
        start_date = '2016-01-05'
        end_date = '2016-01-05'
        currency = 'USD'
        mac_database = db.MacDatabase(cu.create_config_file_path(), environment=environment)
        argument = cu.ExtractionArgument(start_date=start_date, end_date=end_date, currency=currency,
                                         database=mac_database)
        corp_extract = cu.CorpUniverseTableExtraction(
            cu.ExtractionArgument(start_date, end_date, currency, mac_database))
        df_corp_quotes = corp_extract.extract_corp_universe()
        calculator = cu.CorpSpreadCalculator(df_corp_quotes, clip_outliers=False, environment=environment)
        actual_count = len(df_corp_quotes)
        expected_count = 25306
        self.assertEqual(actual_count, expected_count)







