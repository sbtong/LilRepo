----------  Running sovereign derivation process through command line -----------------

Sovereign derivation depends on two steps: Filtration and Bootstrapping

Please note you must change directory so macpy is the working directory.
Filtration depends on oracle tns set to

To run the filtration code, see below example:
## start date = 2016-08-22
## end date = 2016-08-22
## country = CO -- Columbia
Drive:\ContentDev-Mac\macpy>python runBondFilter.py -e DEV CO 2016-08-22 2016-08-22 -l log.config -t

Running the above command lines populates the filtered bonds table for the specified date range.
You can test whether results were written by running the below sql:
###DEV (dev_mac_db)

USE MarketData
SELECT cv.CurveShortName, fb.*
FROM DerivCurveFilteredBond fb
JOIN Curve cv on fb.CurveId = cv.CurveId
WHERE TradeDate = '2016-08-22'
AND cv.CurveShortName = 'CO.COP.GVT.ZC'

You can verify the update occurred by checking whether the Lub (user) and Lud (date/time update occurred) are expected

----------------------------------------------------------------------------------------------------------------------
To run the bootstrapping code, see below example:
## start date = 2016-08-22
## end date = 2016-08-22
## country = CO -- Columbia
Drive:\ContentDev-Mac\macpy>python bootstrapper.py -s 2016-08-22 -e 2016-08-22 -c COP -d DEV


USE MarketData
SELECT *
FROM ResearchCurves
WHERE TradeDate = '2016-08-22'
AND CurveName = 'CO.COP.GVT.ZC'


Note that adding -t to command line writes results to CurveNodeQuote table
Drive:\ContentDev-Mac\macpy>python bootstrapper.py -s 2016-08-22 -e 2016-08-22 -c COP -t -d DEV

USE MarketData
SELECT TradeDate, cq.Lud, Cq.Lub
FROM Curve cv
JOIN CurveNodes cn on cn.CurveId = cv.CurveId
JOIN CurveNodeQuote cq on cn.CurveNodeId = cq.CurveNodeId
WHERE cq.TradeDate >= '2016-08-22'
AND cv.CurveShortName = 'CO.COP.GVT.ZC'


----------------------------------------------------------------------------------------------------------------------
Bond Corrections

SELECT top 1 *
FROM DerivCurveFilteredBondCorrection
WHERE ItemId=1


INSERT INTO [dbo].[DerivCurveFilteredBondCorrection] (CurveId, TradeDate, InstrCode, ItemId, ItemValue, Description, Lud, Lub)
VALUES(200302047,'2014-01-01',1998857, 14, 0, 'Ban Forward', Getdate(), 'hmiao')

INSERT INTO [dbo].[DerivCurveFilteredBondCorrection] (CurveId, TradeDate, InstrCode, ItemId, ItemValue, Description, Lud, Lub)
VALUES(200302071,'2016-03-10',1056473, 1, -1, 'Suspect Price', Getdate(), 'hmiao')
