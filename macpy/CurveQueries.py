from utils import Utilities
import pandas as pd
import pandas.io.sql as sql
import logging

# pandas_version = int(pd.version.version.split('.')[1])
pandas_version = int(pd.__version__.split('.')[1])

############################################## Curve Queries generator#############################################################################
class Curve_Database_Queries(object):
    def __init__(self, mktDBInfo, modelDBInfo, macDBInfo, use_node_quote_final=False, use_research_table = False):
        self.use_node_quote_final = use_node_quote_final
        self.use_research_table = use_research_table
        self.regenerate_currency_rating_curve = False
        self.mktDBInfo = mktDBInfo
        self.modelDBInfo = modelDBInfo
        self.macDBInfo = macDBInfo
        self.macDBConn = Utilities.createMSSQLConnection(self.macDBInfo)
        self.oracle_mkt_conn = Utilities.createOracleConnection(self.mktDBInfo)
        self.oracle_mdl_conn = Utilities.createOracleConnection(self.modelDBInfo)
        
        self.curve_quotes_by_curveshortname = lambda curve_short_name,\
                                                     trade_start_date,\
                                                     trade_end_date:r"""exec [MarketData].[dbo].[DerivCurveYieldCurveTS] 
                                                                        @CurveShortName=%s, 
                                                                        @LookBackStartDate=%s, 
                                                                        @AsOfDate=%s,
                                                                        @useFinalTable=%d"""%("'"+curve_short_name+"'",
                                                                                         "'"+trade_start_date+"'", 
                                                                                         "'"+trade_end_date+"'", 1 if self.use_node_quote_final is True else 0)
        
        self.country_code_to_currency = lambda country_code,trade_start_date,trade_end_date:r"""SELECT currency_code FROM RMG_CURRENCY rc
                                                                                                JOIN risk_model_group rmg ON rmg.rmg_id=rc.rmg_id
                                                                                                WHERE rmg.mnemonic=%s AND rc.from_dt<=%s 
                                                                                                AND rc.thru_dt>%s"""%("'"+country_code+"'", "'"+trade_start_date+"'", 
                                                                                                                      "'"+trade_end_date+"'")

        self.country_code_to_currency_mssql = lambda country_code: r""" SELECT distinct(CurrencyEnum) from [MarketData].[dbo].[Curve] where CountryEnum = '%s'"""%(country_code)
        
        self.curve_detail_by_shortname = lambda curve_short_name:r"""select cv.CountryEnum,cv.CurrencyEnum,
                                                                cv.CurveTypeEnum,cv.CompositeRatingEnum,
                                                                cv.CurveShortName,cv.CurveId,cv.GicsEnum,cv.UnderlyingCurveId, cv.IssuerId, cv.DSFutContrCode,
                                                                cv.UnderlyingTypeEnum, cv.CurveLongName, cv.RIC, cv.ActiveToDate, cv.ActiveFromDate
                                                                from [MarketData].[dbo].[Curve] cv
                                                                where cv.CurveShortName=%s """%("'"+curve_short_name+"'")
        
        self.curve_detail_by_curve_id = lambda curve_id:r"""select cv.CountryEnum,cv.CurrencyEnum,cv.CurveTypeEnum,
                                                                   cv.CompositeRatingEnum,cv.CurveShortName,cv.CurveId,
                                                                   cv.GicsEnum,cv.UnderlyingCurveId, cv.IssuerId
                                                                   from [MarketData].[dbo].[Curve] cv 
                                                                   where cv.CurveId=%d"""%(curve_id)
        
        self.curve_detail_by_country = lambda country_code, trade_start_date, trade_end_date:r"""
        select cv.CountryEnum,cv.CurrencyEnum,cv.CurveTypeEnum, cv.CompositeRatingEnum,cv.CurveShortName,cv.CurveId, cv.GicsEnum,cv.UnderlyingCurveId, cv.IssuerId
        from [MarketData].[dbo].[Curve] cv
        where cv.RegionEnum is NULL and cv.CountryEnum='%s'
        """%(country_code)

        self.curve_detail_by_currency = lambda currency_code, trade_start_date, trade_end_date:r"""
        select cv.CountryEnum,cv.CurrencyEnum,cv.CurveTypeEnum,cv.CompositeRatingEnum,cv.CurveShortName,cv.CurveId,cv.GicsEnum,cv.UnderlyingCurveId, cv.IssuerId
        from [MarketData].[dbo].[Curve] cv
        where cv.RegionEnum is NULL and cv.CurrencyEnum='%s'
        """%(currency_code)

        self.curve_detail_by_curvetypes_country = lambda curve_types, country_code, trade_start_date, trade_end_date:r"""
        select cv.CountryEnum,cv.CurrencyEnum,cv.CurveTypeEnum,cv.CompositeRatingEnum,cv.CurveShortName,cv.CurveId,cv.GicsEnum,cv.UnderlyingCurveId, cv.IssuerId
        from [MarketData].[dbo].[Curve] cv 
        where cv.CurveTypeEnum in (%s) and cv.CountryEnum='%s'"""%(','.join(["'"+c+"'" for c in curve_types]), country_code)


        self.curve_detail_by_curvetypes = lambda curve_types, trade_start_date,trade_end_date:r"""
        select cv.CountryEnum,cv.CurrencyEnum,cv.CurveTypeEnum,cv.CompositeRatingEnum,cv.CurveShortName,cv.CurveId,cv.GicsEnum,cv.UnderlyingCurveId, cv.IssuerId 
        from [MarketData].[dbo].[Curve] cv 
        where cv.CurveTypeEnum in (%s) """%(','.join(["'"+c+"'" for c in curve_types]))

        self.curve_detail_by_curvetypes_currency = lambda curve_types, currency_code, trade_start_date,trade_end_date:r"""
        select cv.CountryEnum,cv.CurrencyEnum,cv.CurveTypeEnum,cv.CompositeRatingEnum,cv.CurveShortName,cv.CurveId,cv.GicsEnum,cv.UnderlyingCurveId, cv.IssuerId 
        from [MarketData].[dbo].[Curve] cv 
        where cv.CurveTypeEnum in (%s) and cv.CurrencyEnum='%s'"""%(','.join(["'"+c+"'" for c in curve_types]), currency_code)

        self.zero_curve_by_currency = lambda currency_code, country_code, trade_start_date, trade_end_date:r"""
        select cv.CountryEnum,cv.CurrencyEnum,cv.CurveTypeEnum,cv.CompositeRatingEnum,cv.CurveShortName,cv.CurveId
        from [MarketData].[dbo].[Curve] cv
        where CurveTypeEnum = 'Sov.Zero'
        and cv.CurrencyEnum='%s'
        and cv.CountryEnum='%s'
        """%(currency_code, country_code)

        self.curve_quotes_query_template = lambda trade_start_date, trade_end_date, curve_short_name:"""
        select CurveShortName, TradeDate, te.InYears, Quote
        from [MarketData].[dbo].[Curve] cv
        join [MarketData].[dbo].[CurveNodes] cn on cv.CurveId = cn.CurveId
        join [MarketData].[dbo].[TenorEnum] te on cn.TenorEnum = te.TenorEnum
        join [MarketData].[dbo].[CurveNodeQuote] cq on cq.CurveNodeId = cn.CurveNodeId
        where
        cq.TradeDate >= '{}' and
        cq.TradeDate <= '{}' and
        cv.CurveShortName = '{}'
        """.format(trade_start_date, trade_end_date, curve_short_name)

        self.zero_curve_by_country = lambda country_code, currency_code: r"""
        select cv.CountryEnum,cv.CurrencyEnum,cv.CurveTypeEnum,cv.CompositeRatingEnum,cv.CurveShortName,cv.CurveId 
        from [MarketData].[dbo].[Curve] cv 
        where CurveTypeEnum = 'Sov.Zero' 
        and cv.CountryEnum='%s'
        and cv.CurrencyEnum='%s'"""%(country_code, currency_code)

        self.curve_data_by_country = lambda country_code, curve_code, curve_sub_code: r"""
        select * from [EJV_rigs].[dbo].[curves] cv
        where cv.cntry_cd = %s
        and cv.curve_cd = %s
        and cv.ayc_fl = 'y'
        and cv.curve_sub_cd = %s
        and cv.status_cd = 'AOK'""" % ("'"+country_code+"'", "'"+curve_code+"'", "'"+curve_sub_code+"'")
        
        self.swap_quotes = lambda chain_ric, country_code, currency_code, trade_start_date, trade_end_date: """
        Declare @tradeDateBegin  date = '%s',
                @tradeDateEnd  date = '%s',
                @countryCode   varchar(2) = '%s',
				@chainRic      varchar(30) = '%s',
				@currencyEnum   varchar(3) = '%s'
        -- To avoid cross server joins, we create a temp table from PROD_VNDR_DB and join Points table against the temp table
		SELECT * INTO #t_swapquotes
		FROM [PROD_VNDR_DB].DataScopeSelect.dbo.TRCurveNodeTS q
		WHERE
		yield != -9999402 AND yield != -9999401
        AND [close price] != -9999402 and [close price] != -9999401
		AND q.[trade date] >= @tradeDateBegin AND q.[trade date] <= @tradeDateEnd
		AND q.ric like '%%'+@currencyEnum+'%%'

        SELECT [trade date] as TradeDate,  [close price] as DiscFactor,
        yield as Quote, p.rt_ric as Ric, te.InYears as InYears, p.short_name
        FROM ejv_rigs..curves c
        JOIN ejv_rigs..points p on p.curve_id = c.curve_id AND c.curve_sub_cd = 'SWPZ'
        AND c.cntry_cd = @countryCode AND c.status_cd in ('AOK','PEN')
        JOIN #t_swapquotes q on q.ric = p.rt_ric
		LEFT JOIN TenorEnum te ON te.TenorEnum = p.short_name
        WHERE c.chain_ric = @chainRic
       -- AND (te.InYears > .20) --exclude short-end tenors for extreme vol
        --order by [trade date], te.InYears""" % (trade_start_date, trade_end_date, country_code, chain_ric, currency_code)

        self.ois_swap_quotes = lambda chain_ric, country_code, currency_code, trade_start_date, trade_end_date: """
        Declare @tradeDateBegin  date = '%s',
                @tradeDateEnd  date = '%s',
                @countryCode   varchar(2) = '%s',
        		@chainRic      varchar(30) = '%s',
        		@currencyEnum   varchar(3) = '%s'
              -- To avoid cross server joins, we create a temp table from PROD_VNDR_DB and join Points table against the temp table
        SELECT * INTO #t_swapquotes
        FROM [PROD_VNDR_DB].DataScopeSelect.dbo.[TR_OIS_EOD_ZeroCurveNode] q
        WHERE
        yield != -9999402 AND yield != -9999401 
        AND [close price] != -9999402 and [close price] != -9999401
        AND q.[trade date] >= @tradeDateBegin AND q.[trade date] <= @tradeDateEnd
        AND q.ric like '%%'+@currencyEnum+'%%'
              SELECT [trade date] as TradeDate,  [close price] as DiscFactor,
              yield as Quote, p.VendorDataInternalId as Ric, te.InYears as InYears, p.NodeShortName
              FROM [MarketData].dbo.OvernightIndexSwap c
              JOIN [MarketData].dbo.OvernightIndexSwapNodes p on p.OISID = c.OISID
              AND c.CurrencyEnum = @currencyEnum
              JOIN #t_swapquotes q on q.ric = p.VendorDataInternalId
         LEFT JOIN TenorEnum te ON  p.TenorEnum = te.TenorEnum
              WHERE c.OISShortName = @chainRic
             """ % (trade_start_date, trade_end_date, country_code, chain_ric, currency_code)

        self.fwd_swap_quotes = lambda fixing_tenor, country_code, currency_code, trade_start_date, trade_end_date: """
        Declare @tradeDateBegin  date = '%s',
                @tradeDateEnd  date = '%s',
                @countryCode   varchar(2) = '%s',
				@currencyEnum   varchar(3) = '%s',
				@fixingTenor   varchar(10) = '%s'
        -- To avoid cross server joins, we create a temp table from PROD_VNDR_DB and join FRANodes against the temp table
		SELECT * INTO #t_fwdswapquotes
		FROM [PROD_VNDR_DB].DataScopeSelect.dbo.TRCurveNodeTS q
		WHERE
		yield != -9999402 AND yield != -9999401
        AND [close price] != -9999402 and [close price] != -9999401
		AND q.[trade date] >= @tradeDateBegin AND q.[trade date] <= @tradeDateEnd
		AND q.ric like '%%'+@currencyEnum+'%%'

        SELECT [trade date] as TradeDate,  [close price] as DiscFactor,
        yield as Quote, q.RIC, te.InYears as InYears, te.TenorEnum
        FROM [MarketData].dbo.FRA fwd
        JOIN [MarketData].dbo.FRANodes fn on fwd.FRAId = fn.FRAId
        JOIN #t_fwdswapquotes q on q.ric = fn.VendorDataInternalID
		LEFT JOIN TenorEnum te ON te.TenorEnum = fn.TenorEnum
        WHERE fwd.Currency = @currencyEnum and fwd.FixingTenor = @fixingTenor
       -- AND (te.InYears > .20) --exclude short-end tenors for extreme vol
        --order by [trade date], te.InYears""" % (trade_start_date, trade_end_date, currency_code, currency_code, fixing_tenor)

        self.curve_nodes = lambda curve_id: r"""
            SELECT cv.CurveTypeEnum, cv.CurveShortName, te.InYears, cn.TenorEnum
            FROM [MarketData]..[Curve] cv
            JOIN [MarketData]..[CurveNodes] cn on cv.Curveid = cn.Curveid
            JOIN [MarketData]..[TenorEnum] te on cn.TenorEnum = te.TenorEnum
            WHERE cv.Curveid = %s""" % (curve_id)

        self.curve_by_country_quotes = lambda curve_id, trade_start_date, trade_end_date:r"""
            select date, spot, term
            from [EJV_ejv_common].[dbo].[analytic_yield_curve_hist]
            where curve_id =
            (select convert(VARBINARY(8),cast(%d as bigint)))
            and date >= %s
            and date <= %s"""%(curve_id,"'"+trade_start_date+"'", "'"+trade_end_date+"'")

        self.curve_node_quotes = lambda curve_node_id, trade_start_date, trade_end_date:(r"""
            select Quote,TradeDate from [MarketData].[dbo].[CurveNodeQuote]
            where CurveNodeId=%d
            and TradeDate >= %s
            and TradeDate <= %s"""%(curve_node_id,
                                    "'"+trade_start_date+"'",
                                    "'"+trade_end_date+"'")) if self.use_node_quote_final == True else(
                                                            (r"""
                                                                select Quote,TradeDate from [MarketData].[dbo].[CurveNodeQuote]
                                                                where CurveNodeId=%d
                                                                and TradeDate >= %s
                                                                and TradeDate <= %s"""%(curve_node_id,
                                                            "'"+trade_start_date+"'", "'"+trade_end_date+"'"))
                                                                                            )
        self.derived_data_corp_bond_price_currency = lambda currency_code, trade_start_date,\
                                                     trade_end_date, IssuerId, IssTypeCode, OrgTypeCode, SnrtyCode:r"""
         exec [MarketData].[dbo].[DerivedDataCorpBondPriceForCcy]
                            @tradeDateBegin = '%s',
                            @tradeDateEnd = '%s',
                            @currencyISO = '%s',                            
                            @IssuerId = %s,
                            @IssTypeCode = %s,
                            @OrgTypeCode = %s,
                            @SnrtyCode = %s"""%(trade_start_date, trade_end_date, currency_code,
                                                  str(IssuerId) if IssuerId is not None else 'null',
                                                  "'"+IssTypeCode+"'" if IssTypeCode is not None else 'null',
                                                  "'"+OrgTypeCode+"'" if OrgTypeCode is not None else 'null',
                                                  "'"+SnrtyCode+"'" if SnrtyCode is not None else 'null')

        self.derived_data_corp_bond_price_currency_rating = lambda currency_code, trade_start_date, trade_end_date: r"""
                exec [MarketData].[dbo].[DerivedDataCorpBondPriceForCcyWithRating]
                    @tradeDateBegin = '%s',
                    @tradeDateEnd = '%s',
                    @IssuerId = null,
                    @currencyISO = '%s',
                    @IssTypeCode = 'CORP',
                    @OrgTypeCode = null,
                    @RatingEnum = null,
                    @IncCallables = 'Y',
                    @IncPuttables = 'N',
                    @SnrtyCode = null
                    """ % (trade_start_date, trade_end_date, currency_code)

        self.derived_swap_yield = lambda country_code, trade_start_date, trade_end_date: r"""
            Declare @tradeDateBegin  date = '%s',
            @tradeDateEnd  date = '%s',
            @countryCode   varchar(2) = '%s'
            select cv.CurveId, CurveShortName, TradeDate, te.InYears, Quote
            from  [MarketData].[dbo].[Curve] cv
            join [MarketData].[dbo].[CurveNodes] cn on cv.CurveId = cn.CurveId
            join MarketData.dbo.TenorEnum te on cn.TenorEnum = te.TenorEnum
            join [MarketData].[dbo].CurveNodeQuote cq on cq.CurveNodeId = cn.CurveNodeId
            where
            (cq.TradeDate >= @tradeDateBegin and
            cq.TradeDate <= @tradeDateEnd and
            cv.CountryEnum = @CountryCode ) and
            (cv.CurveTypeEnum = 'SwapZrSpr' or
            cv.CurveTypeEnum = 'Sov.Zero')
            """ % (trade_start_date, trade_end_date, country_code)

        self.derived_irswap_yield = lambda country_code, currency_code, trade_start_date, trade_end_date: r"""
            Declare @tradeDateBegin  date = '%s',
            @tradeDateEnd  date = '%s',
            @countryCode   varchar(2) = '%s',
            @currencyCode  varchar(3) = '%s'
            select cv.CurveId, CurveShortName, TradeDate, te.InYears, Quote
            from  [MarketData].[dbo].[Curve] cv
            join [MarketData].[dbo].[CurveNodes] cn on cv.CurveId = cn.CurveId
            join MarketData.dbo.TenorEnum te on cn.TenorEnum = te.TenorEnum
            join [MarketData].[dbo].CurveNodeQuote cq on cq.CurveNodeId = cn.CurveNodeId
            where
            cq.TradeDate >= @tradeDateBegin and
            cq.TradeDate <= @tradeDateEnd and
            cv.CountryEnum = @countryCode and
            cv.CurrencyEnum = @currencyCode and
            cv.CurveTypeEnum = 'SwapZC'
            ORDER BY TradeDate, InYears
            """ % (trade_start_date, trade_end_date, country_code, currency_code)

        self.derived_data_corp_bond_price_issuer = lambda currency_code, trade_start_date,\
                                                     trade_end_date, curve_short_name:r"""exec [MarketData].[dbo].[DerivedDataCorpBondPriceForCcy_Issuer]
                                                                        @tradeDateBegin = %s,
                                                                        @tradeDateEnd = %s,
                                                                        @currencyISO = %s,
                                                                        @curveShortName = %s
                                                                        """%("'"+trade_start_date+"'",
                                                                                              "'"+trade_end_date+"'",
                                                                                              "'"+currency_code+"'",
                                                                                              "'"+curve_short_name+"'")

        self.derived_data_corp_bond_price_abs = lambda currency_code, trade_start_date,\
                                                     trade_end_date, DebtIssTypeCode, topCompositeGradeIncl, bottomCompositeGradeIncl:r"""exec [MarketData].[dbo].[DerivedDataABSFieldMapNew]
                                                                        @tradeDateBegin = %s,
                                                                        @tradeDateEnd = %s,
                                                                        @currencyISO = %s,
                                                                        @debtIssTypeCode = %s,
                                                                        @topCompositeGradeIncl = %s,
                                                                        @bottomCompositeGradeIncl = %s"""%("'"+trade_start_date+"'",
                                                                                              "'"+trade_end_date+"'",
                                                                                              "'"+currency_code+"'",
                                                                                              "'"+DebtIssTypeCode+"'",
                                                                                              "'"+topCompositeGradeIncl+"'" if topCompositeGradeIncl is not None else 'null',
                                                                                              "'"+bottomCompositeGradeIncl+"'" if bottomCompositeGradeIncl is not None else 'null')

        self.derived_data_corp_bond_term_structure = lambda currency_code, country_code, \
                                            trade_start_date,trade_end_date,\
                                            debt_iss_type_code, snrty_code, BktVectorBegin,\
                                            BktVectorEnd,\
                                            BktVectorTenor:r"""exec MarketData.[dbo].[DerivedDataCorpBondYieldTermStructure]
                                                               @tradeDateBegin = %s, @tradeDateEnd = %s,
                                                               @currencyISO = %s, @countryISO = %s,
                                                               @DebtIssTypeCode = %s, @SnrtyCode = %s,
                                                               @BktVectorBegin = %s, @BktVectorEnd = %s,
                                                               @BktVectorTenor = %s"""%("'"+trade_start_date+"'",
                                                                                         "'"+trade_end_date+"'",
                                                                                         "'"+currency_code+"'",
                                                                                         "'"+country_code+"'",
                                                                                         "'"+debt_iss_type_code+"'",
                                                                                         "'"+snrty_code+"'",
                                                                                         "'"+BktVectorBegin+"'",
                                                                                         "'"+BktVectorEnd+"'",
                                                                                         "'"+BktVectorTenor+"'")

        self.derived_data_sovereign_curve_yields = lambda trade_date, currency_code, priCtryFlag, priCurFl:r"""exec [MarketData].[dbo].[DerivedDataGetSovereignCurveYieldsEMG]
                                                                                                               @TradeDate = %s, @ISOCurCode = %s, @PriCtryFl = %d,
                                                                                                               @PriCurFl = %s"""%("'"+trade_date+"'",
                                                                                                                                  "'"+currency_code+"'",
                                                                                                                                  priCtryFlag,
                                                                                                                                  "'"+str(priCurFl)+"'" if priCurFl is not None else 'null'
                                                                                                                                  )

        self.emg_CDS_spreads = lambda trade_date, currency_code:r"""declare @CountryEnumTable table (Name varchar(100), CountryEnum varchar(100), CurrencyEnum varchar(100))
																	insert into @CountryEnumTable
																	(
																		Name,
																		CountryEnum,
																		CurrencyEnum
																	)
																	SELECT
																		  [Name]
																		  ,[CountryEnum]
																		  ,[CurrencyEnum]
																	  FROM [Metadata].[dbo].[CountryEnum]
																	insert into @CountryEnumTable
																	values ('Venezuela', 'VE', 'VEF')
																	SELECT [Date]
																		  ,cn.CountryEnum
																		  ,cn.CurrencyEnum
																		  ,[Spread5y]
																		  ,[Spread10y]
																		  ,[Spread30y]
																	  FROM [Markit].[dbo].[MarkitCdsSpreads]
																	  left join @CountryEnumTable cn on cn.Name = Country
																	  where
																	  Ticker in
																	('BRAZIL',
																	'ARGENT',
																	'CHILE',
																	'CHINA',
																	'COLOM',
																	'CROATI',
																	'HUNGAA',
																	'IGB',
																	'INDON',
																	'JAMAN',
																	'LEBAN',
																	'MALAYS',
																	'MEX',
																	'PAKIS',
																	'PANAMA',
																	'PERU',
																	'PHILIP',
																	'POLAND',
																	'ROMANI',
																	'RUSSIA',
																	'SOAF',
																	'TURKEY',
																	'URUGAY',
																	'VENZ',
																	'UKRAIN')
                                                                    and Date = %s
                                                                    and cn.CurrencyEnum = %s
                                                                    AND Sector = 'Government'
                                                                    and Ccy = 'USD'
                                                                    and DocClause like '%%CR%%'
                                                                    AND Tier = 'SNRFOR'
                                                                    """%("'"+trade_date+"'", "'"+currency_code+"'")



        self.futures_price_contract_code = lambda futures_contract_code, trade_start_date, trade_end_date:r"""select Date_, Settlement, qai.dbo.DSFutContrInfo.FutCode, LastTrdDate
                                                                                                             FROM qai.dbo.DSFutContrVal
                                                                                                             JOIN qai.dbo.DSFutContrInfo on qai.dbo.DSFutContrInfo.FutCode=DSFutContrVal.FutCode
                                                                                                             JOIN ( SELECT DISTINCT DSFutContr.ContrCode,DSFutCalcSerInfo.ClsCode
                                                                                                                     FROM qai.dbo.DSFutContr
                                                                                                                      left JOIN (qai.dbo.DSFutClass
                                                                                                                                JOIN qai.dbo.DSFutCalcSerInfo on DSFutCalcSerInfo.ClsCode=DSFutClass.ClsCode) on DSFutClass.ContrCode=DSFutContr.ContrCode)
                                                                                                                as T ON (T.ClsCode IS NULL OR T.ClsCode=DSFutContrInfo.ClsCode) AND T.ContrCode=DSFutContrInfo.ContrCode
                                                                                                                    WHERE
                                                                                                                    T.ContrCode=%d AND
                                                                                                                    Settlement IS NOT NULL and Settlement > 0
                                                                                                                    --and VOLUME is NOT NULL
                                                                                                                    AND DSFutContrInfo.LastTrdDate IS NOT NULL
                                                                                                                    AND LastTrdDate > Date_
                                                                                                                    AND Date_ BETWEEN %s AND %s
                                                                                                                    and ContrDate like '[0-9][0-9][0-9][0-9]'
                                                                                                                    ORDER BY Date_
                                                                                                                --cast(substring(ContrDate, 1, 2) AS INT) AS _MONTH, cast(substring(ContrDate, 3, 2) AS INT) AS DECADE
                                                                                                            """%(futures_contract_code, "'"+trade_start_date+"'",
                                                                                                                 "'"+trade_end_date+"'")

        self.base_correlation = lambda trade_start_date, trade_end_date, curve_long_name: r""" select [Date], [Index Name], [Red Code], [Detachment], [Base Correlation]
                                                                                        from Markit.dbo.MarkitCdsTrancheComposites 
                                                                                        where [Date] >= %s and [Date] <= %s and [index name] like %s
                                                                                        Order by [Date], [Detachment]"""%("'"+trade_start_date+"'", "'"+trade_end_date+"'", "'%"+curve_long_name+"%'")
    
    def fetch_mac_curves_query_result(self, query):
        results = []
        try:
            cursor = self.macDBConn.cursor()
            cursor.execute(query)
            if cursor.description is not None:
                results = cursor.fetchall()
        except Exception as e:
            logging.error(e)
        finally:
            self.macDBConn.commit()
        return results

    def fetch_mac_curves_query_dataframe(self, query):
        result = None
        connection = Utilities.createMSSQLConnection(self.macDBInfo)
        try:
            if pandas_version >= 15:
                result = sql.read_sql(query, connection)
            else:
                result = sql.read_frame(query, connection)
        except Exception as e:
            logging.error(e)
        finally:
            connection.commit()
            connection.close()
        return result

    def fetch_market_query_result(self, query):
        results = []
        try:
            cursor = self.oracle_mkt_conn.cursor()
            cursor.execute(query)
            if cursor.description is not None:
                results = cursor.fetchall()
        except Exception as e:
            logging.error(e)
        finally:
            self.oracle_mkt_conn.commit()
        return results

    def fetch_model_query_result(self, query):
        results = []
        try:
            cursor = self.oracle_mdl_conn.cursor()
            cursor.execute(query)
            if cursor.description is not None:
                results = cursor.fetchall()
        except Exception as e:
            logging.error(e)
        finally:
            self.oracle_mdl_conn.commit()
        return results

query_result_cache = {}

def extract_dataframe(curve_gen, query):
    result = None
    connection = Utilities.createMSSQLConnection(curve_gen.macDBInfo)

    if query in query_result_cache:
        #print 'Cached result', query
        return query_result_cache[query]

    try:
        if pandas_version >= 15:
            result = sql.read_sql(query, connection)
        else:
            result = sql.read_frame(query, connection)
        query_result_cache[query] = result.copy(deep=True)
    finally:
        connection.commit()
        connection.close()

    return result

