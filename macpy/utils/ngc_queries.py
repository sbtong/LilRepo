import macpy.utils.database as db
from string import Template

def get_OAS_data(start_date, end_date, currency):

    sql="""
with latest_issuer as (select *
						   from   MarketData..Issuer i
						   where  FromDate = (select MAX(FromDate)
											  from   MarketData..Issuer i2
											  where  i2.IssuerId = i.IssuerId))
	, OrgCodeToIssuerId as (select ix.IssuerVendorId as OrgCode, ix.IssuerId from [MarketData].[dbo].[IssuerXref] ix
        where          ix.IssuerVendorIdType = 'TREJVOrgCode'
        and		    ix.ToDate = (	select	max(ix2.ToDate)  -- pick alwasy most recent, don't try to retrieve historical org mappings
                                    from	MarketData..IssuerXref ix2
                                    where	ix2.IssuerVendorIdType = 'TREJVOrgCode'
                                    and		ix2.IssuerVendorId = ix.IssuerVendorId))
SELECT
    crv.[AnalysisDate]
    ,crv.[ISIN]
    , crv.OAS_Swap
    , issuer.RiskEntityId
	, RE.CountryOfRisk
	, region.Region
	, region.RegionGroup
    , REissuer.IssuerName -- name of RiskEntity
	, TICKER = RE.Ticker
    , IndGrp.IndustryGroup
    , IndMap.Market
    , IndMap.Sector
    , IndMap.SectorGroup
    , statfld.DebtISOCurCode as Currency
    , statfld.MatDate
    --, statfld.SnrtyCodeDesc
	, sbe.PricingTier
    , statfld.FrstCpnRate as Coupon
    , statfld.CouponPaymentFrequencyCode
    , crv.Effective_Duration
    , AmtOutstanding = sa.value_ / 1e6
    FROM [MarketData].[dbo].[DerivCurveBondStatisticsCombined] crv
    inner join [ejv_views].dbo.[v_Axioma_StaticFields]  statfld on crv.InstrCode = statfld.InstrCode
    left join [QAI].[dbo].[FIEJVOrgInfo]				orgInfo on statfld.IssrGEMOrgId = orgInfo.GemOrgID
    left join OrgCodeToIssuerId							ix		on ix.OrgCode = orgInfo.OrgCode
    left join latest_issuer								issuer	on issuer.IssuerId = ix.IssuerId
	left join [MarketData].[dbo].[RiskEntity]			RE		on RE.RiskEntityId = issuer.RiskEntityId
																and RE.ToDate = '31-Dec-9999'
    left join [MarketData].[dbo].[AL_IssuerIndustryGroup] IndGrp on issuer.RiskEntityId = IndGrp.IssuerId
    left join [MarketData].[dbo].[AL_IndustryMap]		IndMap	on IndGrp.IndustryGroup = IndMap.IndustryGroup
	left join latest_issuer								REissuer on REissuer.IssuerId = issuer.RiskEntityId
	left join [MarketData].[dbo].[AL_CountryToRegionMap]  region on region.iso_scntry_cd = RE.CountryOfRisk
    left join [QAI].dbo.FIEJVSecAmt               sa on    sa.InstrCode = statfld.InstrCode		/* amt out*/
													and  sa.AmtDate = (select max(AmtDate) from [QAI].dbo.FIEJVSecAmt saa
																where saa.InstrCode=sa.InstrCode
																and   saa.Item=sa.Item
																and   saa.AmtDate <= crv.[AnalysisDate])
													and  sa.Item=191
-- *use the following 2 joins once MarketData..Bond is complete to replace AL_SubordinationTier*
	left join MarketData..InstrumentXref  AxId	on AxId.SecurityIdentifier = crv.ISIN
													and AxId.SecurityIdentifierType = 'ISIN'
													and AxId.FromDate <= crv.AnalysisDate and AxId.ToDate >= crv.AnalysisDate
	left join MarketData..Bond					sbt on sbt.AxiomaDataId = AxId.AxiomaDataId
	left join MarketData..SubordinationTierEnum sbe on sbe.SubordinationTierEnum = sbt.SubordinationTier
    --left join [MarketData].[dbo].[BondPrice_Corrections]          pcorr  on    pcorr.InstrCode = crv.InstrCode
    --                                  and  pcorr.[Date] = crv.AnalysisDate

 where statfld.DebtISOCurCode = '%s'
    and crv.AnalysisDate >= '%s'
    and crv.AnalysisDate <= '%s'
    order by [crv].ISIN
    """ % (currency, start_date, end_date)

    spread_df = db.MSSQL.extract_dataframe(sql, environment='DEV')
    return spread_df

def commit_node(AxiomaDataId, TradeDate, TenorEnum, Quote, category):
    delete_query(AxiomaDataId, TradeDate, TenorEnum, category)
    sql_insert = Template("""INSERT INTO [MarketData].[dbo].[xcIssuerCurve]
        (AxiomaDataId, TradeDate, TenorEnum, Quote, category, Lud, Lub)
        VALUES($AxiomaDataId, '$TradeDate', '$TenorEnum', $Quote, '$category', Getdate(), 'dantonio');""")

    sql_commit = sql_insert.substitute(AxiomaDataId=AxiomaDataId,
                                       TradeDate= TradeDate,
                                       TenorEnum=TenorEnum,
                                       Quote = Quote,
                                       category = category)

    db.MSSQL.execute_commit(sql_commit, environment='DEV')

def commit_ModelOAS(AxiomaDataId, TradeDate, ISIN, Lvl, Chg, LvlDiff, ChgDiff, OutlierLvl, OutlierChg):
    delete_ModelOAS(AxiomaDataId, TradeDate, ISIN)
    sql_insert = Template("""INSERT INTO [MarketData].[dbo].[xcIssuerModelOAS]
        (AxiomaDataId, TradeDate, ISIN, Lvl, Chg, LvlDiff, ChgDiff, OutlierLvl, OutlierChg, Lud, Lub)
        VALUES($AxiomaDataId, '$TradeDate', '$ISIN', $Lvl, $Chg, $LvlDiff, $ChgDiff, $OutlierLvl, $OutlierChg, Getdate(), 'dantonio');""")

    sql_commit = sql_insert.substitute(AxiomaDataId=AxiomaDataId,
                                       TradeDate= TradeDate,
                                       ISIN=ISIN,
                                       Lvl = Lvl,
                                       Chg = Chg,
                                       LvlDiff=LvlDiff,
                                       ChgDiff=ChgDiff,
                                       OutlierLvl = OutlierLvl,
                                       OutlierChg = OutlierChg)
    db.MSSQL.execute_commit(sql_commit, environment='DEV')

def delete_ModelOAS(AxiomaDataId, TradeDate, ISIN):
    sqlstatement = Template("""
    DELETE from [MarketData].[dbo].[xcIssuerModelOAS]
    where AxiomaDataId = $AxiomaDataId
    and TradeDate = '$TradeDate'
    and ISIN = '$ISIN'
    """).substitute({'AxiomaDataId':AxiomaDataId, 'TradeDate': TradeDate, 'ISIN':ISIN})
    db.MSSQL.execute_commit(sqlstatement, environment='DEV')

def commit_SumAmtOutstanding(AxiomaDataId, TradeDate, SumAmtOutstanding):
    delete_SumAmtOutstanding(AxiomaDataId, TradeDate)
    sql_insert = Template("""INSERT INTO [MarketData].[dbo].[xcIssuerProperty]
        (AxiomaDataId, TradeDate, SumAmtOutstanding, Lud, Lub)
        VALUES($AxiomaDataId, '$TradeDate', '$SumAmtOutstanding', Getdate(), 'dantonio');""")

    sql_commit = sql_insert.substitute(AxiomaDataId=AxiomaDataId,
                                       TradeDate= TradeDate,
                                       SumAmtOutstanding=SumAmtOutstanding)

    db.MSSQL.execute_commit(sql_commit, environment='DEV')

def delete_SumAmtOutstanding(AxiomaDataId, TradeDate):
    sqlstatement = Template("""
    DELETE from [MarketData].[dbo].[xcIssuerProperty]
    where AxiomaDataId = $AxiomaDataId
    and TradeDate = '$TradeDate'
    """).substitute({'AxiomaDataId':AxiomaDataId, 'TradeDate': TradeDate})
    db.MSSQL.execute_commit(sqlstatement, environment='DEV')


def delete_query(AxiomaDataId, TradeDate, TenorEnum, category):
    sqlstatement = Template("""
    DELETE from [MarketData].[dbo].[xcIssuerCurve]
    where AxiomaDataId = $AxiomaDataId
    and TradeDate = '$TradeDate'
    and TenorEnum = '$TenorEnum'
    and category = '$category'
    """).substitute({'AxiomaDataId':AxiomaDataId, 'TradeDate': TradeDate, 'TenorEnum':TenorEnum, 'category':category})
    db.MSSQL.execute_commit(sqlstatement, environment='DEV')

def get_curve(CurveName, Category,TradeDate):
    sqlstatement = Template("""
    select TimeInYears, Level from marketdata.dbo.ResearchCurves
    where CurveName = '$curvename'
    and Category = '$Category'
    and TradeDate = '$TradeDate'
    order by TimeInYears
    """).substitute({'curvename':CurveName, 'TradeDate': TradeDate, 'Category':Category})

    curve_i = db.MSSQL.extract_dataframe(sqlstatement, environment='DEV')
    return curve_i

# def get_curveId(IssuerId, Currency,PricingTier, TaxStatus=None, BespokeProperty=None):
#     sqlstatement = Template("""
#       SELECT AxiomaDataId
#       FROM [MarketData].[dbo].[xcIssuerCurveUniverse]
#       where IssuerId = $IssuerId
#       and CurrencyEnum = '$Currency'
#       and PrcTierEnum = '$PricingTier'
#     """).substitute({'IssuerId':IssuerId, 'Currency': Currency, 'PricingTier':PricingTier})
#     CurveId = db.MSSQL.extract_dataframe(sqlstatement, environment='DEV')
#     return CurveId

def get_curveId(CurveShortName, RiskEntityId, Currency, PricingTier):
    sqlstatement = Template("""
      exec MarketData..xcUpdateIssuerCurveUniverse @CurveShortName='$CurveShortName', @RiskEntityId=$RiskEntityId, @Currency='$Currency',  @PricingTier='$PricingTier'
    """).substitute({'CurveShortName':CurveShortName, 'RiskEntityId': RiskEntityId, 'Currency':Currency, 'PricingTier':PricingTier})
    CurveId = db.MSSQL.extract_dataframe(sqlstatement, environment='DEV')
    return CurveId.AxiomaDataId.values[0]

def get_max_curveId():
    sql = """select MAX(CurveId) from [MarketData].[dbo].[DA_CurveUniverse]"""
    max_curveId = db.MSSQL.extract_dataframe(sql, environment='DEV')
    max = max_curveId.values[0][0]
    return max

def create_curveId(CurveId, RiskEntityId, Currency,PricingTier):
    sql_insert = Template("""INSERT INTO [MarketData].[dbo].[DA_CurveUniverse]
        (CurveId, RiskEntityId, PrcTierEnum, CurrencyEnum, Lud, Lub)
        VALUES($CurveId,$RiskEntityId,'$PricingTier','$Currency', Getdate(), 'dantonio');""")
    sql_commit = sql_insert.substitute(CurveId=CurveId,
                                       RiskEntityId= RiskEntityId,
                                       PricingTier=PricingTier,
                                       Currency = Currency)
    db.MSSQL.execute_commit(sql_commit, environment='DEV')

def get_cluster_curves(Currency,IndustryLevel,Region):
    sql = """
  select * from
    [MarketData].[dbo].[AL_Cluster]
    where
    Currency = '%s'
    and Industry = '%s'
    and Region = '%s'
    """ % (Currency, IndustryLevel, Region)
    cluster_curve = db.MSSQL.extract_dataframe(sql, environment='DEV')
    return cluster_curve

def get_industry_hierarchy(IndustryGroup):
    sql = """
        select * from
        MarketData..AL_IndustryMap
        where IndustryGroup = '%s'
    """ % (IndustryGroup)
    cluster_curve = db.MSSQL.extract_dataframe(sql, environment='DEV')
    return cluster_curve

def get_industry_map_description(IndustryCode):
    sql = """
    SELECT [IndustryDescription]
    FROM [MarketData].[dbo].[AL_IndustryMapDescription]
    where IndustryCode = '%s'
        """ % str(IndustryCode)
    IndMapDesc = db.MSSQL.extract_dataframe(sql, environment='DEV')
    return IndMapDesc['IndustryDescription'].values[0]

def get_cluster_issuer_property(AnalysisDate): #this pulls all issuers, across all currencies
    sql = """
    use MarketData
    declare	@AnalysisDate datetime = '%s';
    --declare @Currency varchar(3) = 's'

 select	u.AxiomaDataId, u.CurveShortName, u.Currency, u.PricingTier,
    re.RiskEntityId, re.Ticker
    , rrMD.RatingRank as MDRank
    , rrSP.RatingRank as SPRank
    ,case when	rerMD.DomesticCurrency = u.Currency then coalesce(rerMD.DomesticIssuerRating, rerMD.DomesticBondImpliedRating, rerMD.ForeignIssuerRating, rerMD.ForeignBondImpliedRating)
    else	coalesce(rerMD.ForeignIssuerRating, rerMD.ForeignBondImpliedRating, rerMD.DomesticIssuerRating, rerMD.DomesticBondImpliedRating)
    end	as MDRating
    ,case when	rerSP.DomesticCurrency = u.Currency then coalesce(rerSP.DomesticIssuerRating, rerSP.DomesticBondImpliedRating, rerSP.ForeignIssuerRating, rerSP.ForeignBondImpliedRating)
    else	coalesce(rerSP.ForeignIssuerRating, rerSP.ForeignBondImpliedRating, rerSP.DomesticIssuerRating, rerSP.DomesticBondImpliedRating)
    end	as SPRating
    , reg.Region
    , reg.RegionGroup
    , IndGrp.IndustryGroup
    , IndMap.Market
    , IndMap.Sector
    , IndMap.SectorGroup
    , IndMap.AllSectors
	, ao.SumAmtOutstanding
    from xcIssuerCurveUniverse	u
    left join RiskEntity			re		on re.RiskEntityId = u.RiskEntityId
                                            and re.ToDate = '31 Dec 9999'
    left join	AL_RiskEntityRating	 rerMD	on	rerMD.RiskEntityId = re.RiskEntityId
                                            and	rerMD.PricingTier = u.PricingTier
                                            and	rerMD.RatingAgency = 'MD'
                                            and rerMD.DateFrom <= @AnalysisDate and rerMD.DateTo >= @AnalysisDate
    left join AL_RatingEnum	rrMD			on	rrMD.Rating = (case when	rerMD.DomesticCurrency = u.Currency then coalesce(rerMD.DomesticIssuerRating, rerMD.DomesticBondImpliedRating, rerMD.ForeignIssuerRating, rerMD.ForeignBondImpliedRating)
                                                                else	coalesce(rerMD.ForeignIssuerRating, rerMD.ForeignBondImpliedRating, rerMD.DomesticIssuerRating, rerMD.DomesticBondImpliedRating)
                                                                end)
                                            and rrMD.RatingAgency = 'MD' and rrMD.RatingScale = 'MLT'
    left join	AL_RiskEntityRating	 rerSP	on	rerSP.RiskEntityId = re.RiskEntityId
                                            and	rerSP.PricingTier = u.PricingTier
                                            and	rerSP.RatingAgency = 'SP'
                                            and rerSP.DateFrom <= @AnalysisDate and rerSP.DateTo >= @AnalysisDate
    left join AL_RatingEnum	rrSP			on	rrSP.Rating = (case when	rerSP.DomesticCurrency = u.Currency then coalesce(rerSP.DomesticIssuerRating, rerSP.DomesticBondImpliedRating, rerSP.ForeignIssuerRating, rerSP.ForeignBondImpliedRating)
                                                                else	coalesce(rerSP.ForeignIssuerRating, rerSP.ForeignBondImpliedRating, rerSP.DomesticIssuerRating, rerSP.DomesticBondImpliedRating)
                                                                end)
                                            and rrSP.RatingAgency = 'SP' and rrSP.RatingScale = 'SLT'
    left join AL_CountryToRegionMap		reg		on reg.iso_scntry_cd = re.CountryOfRisk
    left join AL_IssuerIndustryGroup	IndGrp	on u.RiskEntityId = IndGrp.IssuerId
    left join AL_IndustryMap			IndMap	on IndGrp.IndustryGroup = IndMap.IndustryGroup
	left join xcIssuerProperty		ao on ao.AxiomaDataId = u.AxiomaDataId
	                                and ao.TradeDate = @AnalysisDate

    where (rrMD.RatingRank is not null or rrSP.RatingRank is not null)
    and isnull(rrMD.RatingRank,0) <= 21 and isnull(rrSP.RatingRank,0) <= 21
    --and u.Currency = @Currency
    order by u.AxiomaDataId

    """ % (AnalysisDate)
    df = db.MSSQL.extract_dataframe(sql, environment='DEV')
    return df

def get_cluster_issuer_curves(AnalysisDate):
    sql = """
    select * from MarketData..xcIssuerCurve
    where TradeDate = '%s'
    and category in ( 'll', 'w')
    """ % (AnalysisDate)
    df = db.MSSQL.extract_dataframe(sql, environment='DEV')
    return df

def get_cluster_rating_scale():
    sqlrs = """
    use MarketData
    select RatingRank, Rating from AL_RatingEnum
    where RatingAgency = 'SP' and RatingScale = 'SLT' and IsPrimaryForRank = 1
    and RatingRank <= 21
    """
    df = db.MSSQL.extract_dataframe(sqlrs, environment='DEV')
    return df

def get_cluster_raw_oas_region(AnalysisDate, Currency, industry, industryLevel, region, allSectorFlag):
    sql = """
    use MarketData

    declare	@AnalysisDate datetime = '%s';

    select	u.AxiomaDataId, u.CurveShortName, u.Currency, u.PricingTier,
    re.RiskEntityId, re.Ticker
    , rrMD.RatingRank as MDRank
    , rrSP.RatingRank as SPRank
    ,case when	rerMD.DomesticCurrency = u.Currency then coalesce(rerMD.DomesticIssuerRating, rerMD.DomesticBondImpliedRating, rerMD.ForeignIssuerRating, rerMD.ForeignBondImpliedRating)
    else	coalesce(rerMD.ForeignIssuerRating, rerMD.ForeignBondImpliedRating, rerMD.DomesticIssuerRating, rerMD.DomesticBondImpliedRating)
    end	as MDRating
    ,case when	rerSP.DomesticCurrency = u.Currency then coalesce(rerSP.DomesticIssuerRating, rerSP.DomesticBondImpliedRating, rerSP.ForeignIssuerRating, rerSP.ForeignBondImpliedRating)
    else	coalesce(rerSP.ForeignIssuerRating, rerSP.ForeignBondImpliedRating, rerSP.DomesticIssuerRating, rerSP.DomesticBondImpliedRating)
    end	as SPRating
    , reg.Region
    , reg.RegionGroup
    , IndMap.IndustryGroup
    , IndMap.Market
    , IndMap.Sector
    , IndMap.SectorGroup
    , IndMap.AllSectors
    , dcbs.ISIN
    , dcbs.Effective_Duration
    , dcbs.OAS_Swap
    , AmtOutstanding = sa.value_ / 1e6
    --, *
    from xcIssuerCurveUniverse	u
    left join RiskEntity			re		on re.RiskEntityId = u.RiskEntityId
    and re.ToDate = '31 Dec 9999'
    left join	AL_RiskEntityRating	 rerMD	on	rerMD.RiskEntityId = re.RiskEntityId
    and	rerMD.PricingTier = u.PricingTier
    and	rerMD.RatingAgency = 'MD'
    and rerMD.DateFrom <= @AnalysisDate and rerMD.DateTo >= @AnalysisDate
    left join AL_RatingEnum	rrMD			on	rrMD.Rating = (case when	rerMD.DomesticCurrency = u.Currency then coalesce(rerMD.DomesticIssuerRating, rerMD.DomesticBondImpliedRating, rerMD.ForeignIssuerRating, rerMD.ForeignBondImpliedRating)
    else	coalesce(rerMD.ForeignIssuerRating, rerMD.ForeignBondImpliedRating, rerMD.DomesticIssuerRating, rerMD.DomesticBondImpliedRating)
    end)
    and rrMD.RatingAgency = 'MD' and rrMD.RatingScale = 'MLT'
    left join	AL_RiskEntityRating	 rerSP	on	rerSP.RiskEntityId = re.RiskEntityId
    and	rerSP.PricingTier = u.PricingTier
    and	rerSP.RatingAgency = 'SP'
    and rerSP.DateFrom <= @AnalysisDate and rerSP.DateTo >= @AnalysisDate
    left join AL_RatingEnum	rrSP			on	rrSP.Rating = (case when	rerSP.DomesticCurrency = u.Currency then coalesce(rerSP.DomesticIssuerRating, rerSP.DomesticBondImpliedRating, rerSP.ForeignIssuerRating, rerSP.ForeignBondImpliedRating)
    else	coalesce(rerSP.ForeignIssuerRating, rerSP.ForeignBondImpliedRating, rerSP.DomesticIssuerRating, rerSP.DomesticBondImpliedRating)
    end)
    and rrSP.RatingAgency = 'SP' and rrSP.RatingScale = 'SLT'
    left join AL_CountryToRegionMap		reg		on reg.iso_scntry_cd = re.CountryOfRisk
    left join AL_IssuerIndustryGroup	IndGrp	on u.RiskEntityId = IndGrp.IssuerId
    left join AL_IndustryMap			IndMap	on IndGrp.IndustryGroup = IndMap.IndustryGroup
    left join xcIssuerModelOAS           oasd    on u.AxiomaDataId = oasd.AxiomaDataId
                                               and oasd.TradeDate = @AnalysisDate
                                              and oasd.OutlierLvl = 0
    left join DerivCurveBondStatistics  dcbs    on oasd.ISIN = dcbs.ISIN
                                                and dcbs.AnalysisDate = @AnalysisDate
    left join [QAI].dbo.FIEJVSecAmt               sa on    sa.InstrCode = dcbs.InstrCode		/* amt out*/
                                                and  sa.AmtDate = (select max(AmtDate) from [QAI].dbo.FIEJVSecAmt saa
                                                            where saa.InstrCode=sa.InstrCode
                                                            and   saa.Item=sa.Item
                                                            and   saa.AmtDate <= @AnalysisDate)
                                                and  sa.Item=191

    where (rrMD.RatingRank is not null or rrSP.RatingRank is not null)
    and isnull(rrMD.RatingRank,0) <= 21 and isnull(rrSP.RatingRank,0) <= 21
    and u.Currency = '%s'
    %s and IndMap.%s = '%s'
    and reg.Region = '%s'
    order by AxiomaDataId
    """ % (AnalysisDate, Currency, allSectorFlag, industryLevel, industry, region)
    df = db.MSSQL.extract_dataframe(sql, environment='DEV')
    return df

def get_cluster_raw_oas_global(AnalysisDate, Currency, industry, industryLevel, allSectorFlag):
    sql = """
    use MarketData

    declare	@AnalysisDate datetime = '%s';

    select	u.AxiomaDataId, u.CurveShortName, u.Currency, u.PricingTier,
    re.RiskEntityId, re.Ticker
    , rrMD.RatingRank as MDRank
    , rrSP.RatingRank as SPRank
    ,case when	rerMD.DomesticCurrency = u.Currency then coalesce(rerMD.DomesticIssuerRating, rerMD.DomesticBondImpliedRating, rerMD.ForeignIssuerRating, rerMD.ForeignBondImpliedRating)
    else	coalesce(rerMD.ForeignIssuerRating, rerMD.ForeignBondImpliedRating, rerMD.DomesticIssuerRating, rerMD.DomesticBondImpliedRating)
    end	as MDRating
    ,case when	rerSP.DomesticCurrency = u.Currency then coalesce(rerSP.DomesticIssuerRating, rerSP.DomesticBondImpliedRating, rerSP.ForeignIssuerRating, rerSP.ForeignBondImpliedRating)
    else	coalesce(rerSP.ForeignIssuerRating, rerSP.ForeignBondImpliedRating, rerSP.DomesticIssuerRating, rerSP.DomesticBondImpliedRating)
    end	as SPRating
    , reg.Region
    , reg.RegionGroup
    , IndMap.IndustryGroup
    , IndMap.Market
    , IndMap.Sector
    , IndMap.SectorGroup
    , IndMap.AllSectors
    , dcbs.ISIN
    , dcbs.Effective_Duration
    , dcbs.OAS_Swap
    , AmtOutstanding = sa.value_ / 1e6
    --, *
    from xcIssuerCurveUniverse	u
    left join RiskEntity			re		on re.RiskEntityId = u.RiskEntityId
    and re.ToDate = '31 Dec 9999'
    left join	AL_RiskEntityRating	 rerMD	on	rerMD.RiskEntityId = re.RiskEntityId
    and	rerMD.PricingTier = u.PricingTier
    and	rerMD.RatingAgency = 'MD'
    and rerMD.DateFrom <= @AnalysisDate and rerMD.DateTo >= @AnalysisDate
    left join AL_RatingEnum	rrMD			on	rrMD.Rating = (case when	rerMD.DomesticCurrency = u.Currency then coalesce(rerMD.DomesticIssuerRating, rerMD.DomesticBondImpliedRating, rerMD.ForeignIssuerRating, rerMD.ForeignBondImpliedRating)
    else	coalesce(rerMD.ForeignIssuerRating, rerMD.ForeignBondImpliedRating, rerMD.DomesticIssuerRating, rerMD.DomesticBondImpliedRating)
    end)
    and rrMD.RatingAgency = 'MD' and rrMD.RatingScale = 'MLT'
    left join	AL_RiskEntityRating	 rerSP	on	rerSP.RiskEntityId = re.RiskEntityId
    and	rerSP.PricingTier = u.PricingTier
    and	rerSP.RatingAgency = 'SP'
    and rerSP.DateFrom <= @AnalysisDate and rerSP.DateTo >= @AnalysisDate
    left join AL_RatingEnum	rrSP			on	rrSP.Rating = (case when	rerSP.DomesticCurrency = u.Currency then coalesce(rerSP.DomesticIssuerRating, rerSP.DomesticBondImpliedRating, rerSP.ForeignIssuerRating, rerSP.ForeignBondImpliedRating)
    else	coalesce(rerSP.ForeignIssuerRating, rerSP.ForeignBondImpliedRating, rerSP.DomesticIssuerRating, rerSP.DomesticBondImpliedRating)
    end)
    and rrSP.RatingAgency = 'SP' and rrSP.RatingScale = 'SLT'
    left join AL_CountryToRegionMap		reg		on reg.iso_scntry_cd = re.CountryOfRisk
    left join AL_IssuerIndustryGroup	IndGrp	on u.RiskEntityId = IndGrp.IssuerId
    left join AL_IndustryMap			IndMap	on IndGrp.IndustryGroup = IndMap.IndustryGroup
    left join xcIssuerModelOAS           oasd    on u.AxiomaDataId = oasd.AxiomaDataId
                                               and oasd.TradeDate = @AnalysisDate
                                              and oasd.OutlierLvl = 0
    left join DerivCurveBondStatistics  dcbs    on oasd.ISIN = dcbs.ISIN
                                                and dcbs.AnalysisDate = @AnalysisDate
    left join [QAI].dbo.FIEJVSecAmt               sa on    sa.InstrCode = dcbs.InstrCode		/* amt out*/
                                                and  sa.AmtDate = (select max(AmtDate) from [QAI].dbo.FIEJVSecAmt saa
                                                            where saa.InstrCode=sa.InstrCode
                                                            and   saa.Item=sa.Item
                                                            and   saa.AmtDate <= @AnalysisDate)
                                                and  sa.Item=191

    where (rrMD.RatingRank is not null or rrSP.RatingRank is not null)
    and isnull(rrMD.RatingRank,0) <= 21 and isnull(rrSP.RatingRank,0) <= 21
    and u.Currency = '%s'
    %s and IndMap.%s = '%s'
    order by AxiomaDataId
    """ % (AnalysisDate, Currency, allSectorFlag, industryLevel, industry)
    df = db.MSSQL.extract_dataframe(sql, environment='DEV')
    return df

def get_cluster_hierarchy(IndustryLevel, Industry):
    sql="""
    SELECT top 1 IndustryGroup, Sector, SectorGroup, Market, AllSectors from
    [MarketData].[dbo].[AL_IndustryMap]
    where %s = '%s'
    """ % (IndustryLevel, Industry)
    df = db.MSSQL.extract_dataframe(sql, environment='DEV')
    return df

def NeedXccySupport(Currency, Industry, Region):
    sql="""
    SELECT NeedXccySupport
    FROM [MarketData].[dbo].[AL_Cluster]
    where Currency = '%s'
    and Industry = '%s'
    and Region = '%s'
    """ % (Currency, Industry, Region)
    df = db.MSSQL.extract_dataframe(sql, environment='DEV')
    NeedSupport = df.NeedXccySupport.values[0]
    if NeedSupport == 'Y':
        NeedSupport = True
    else:
        NeedSupport = False
    return NeedSupport

def get_example_curve(AxiomaDataId):
    sql="""
    SELECT t.InYears, c.TenorEnum,
          [Quote]

      FROM [MarketData].[dbo].[xcIssuerCurve] c
      join MarketData..TenorEnum t on  c.TenorEnum = t.TenorEnum
      where AxiomaDataId = '%s'
      and category = 'll'
      order by t.InYears
    """ % (AxiomaDataId)
    df = db.MSSQL.extract_dataframe(sql, environment='DEV')
    return df

def get_supporting_currencies(Currency):
    sql = """
    select  cg2.Currency, cg2.FallbackOnly
    from    MarketData..AL_CurrencyGroup  cg
    join    MarketData..AL_CurrencyGroup  cg2 on cg2.CurrencyGroupNumber = cg.CurrencyGroupNumber
    where   cg.Currency = '%s'
    """ % (Currency)
    df = db.MSSQL.extract_dataframe(sql, environment='DEV')
    #currencyList = df.Currency.astype(str).tolist()
    #return currencyList
    return df

def get_fallback_currency(Currency):
    sql = """
    select  cg2.Currency
    from    MarketData..AL_CurrencyGroup  cg
    join    MarketData..AL_CurrencyGroup  cg2 on cg2.CurrencyGroupNumber = cg.CurrencyGroupNumber
    where   cg.Currency = '%s'
    and             cg2.FallbackOnly = 'y'
    """ % (Currency)
    df = db.MSSQL.extract_dataframe(sql, environment='DEV')
    fallbackCurrency = df.Currency.astype(str).tolist()
    return fallbackCurrency[0]

