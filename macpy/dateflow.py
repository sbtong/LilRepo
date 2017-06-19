import datetime
from dateutil import rrule
import calendar
from dateutil.relativedelta import relativedelta
import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np

class DateFlow(object):
    def __init__(self, coupon_rate, issue, maturity, valuation, freq=2, first_cpn_dt=None, last_cpn_dt=None, legacy=False, notionalAmount = 100.0):
        self.coupon_rate = coupon_rate
        self.freq = freq
        # print first_cpn_dt, last_cpn_dt
        # print issue, maturity, first_cpn_dt, last_cpn_dt
        if isinstance(first_cpn_dt, str):
            self.first_cpn_dt = datetime.datetime.strptime(first_cpn_dt, '%Y-%m-%d')
        else:
             self.first_cpn_dt = first_cpn_dt
        if isinstance(last_cpn_dt, str):
            self.last_cpn_dt = datetime.datetime.strptime(last_cpn_dt, '%Y-%m-%d')
        else:
             self.last_cpn_dt = last_cpn_dt
        if isinstance(issue, str):
            self.issue = datetime.datetime.strptime(issue, '%Y-%m-%d')
        else:
            self.issue = issue
        if isinstance(maturity, str):
            self.maturity = datetime.datetime.strptime(maturity, '%Y-%m-%d')
        else:
            self.maturity = maturity
        if isinstance(valuation, str):
            self.valuation = datetime.datetime.strptime(valuation, '%Y-%m-%d')
        else:
            self.valuation = valuation
        self.period = self.coupon_payment_period(freq)
        self.step = self.coupon_payment_step(freq)
        self.legacy=legacy
        self.notionalAmount = notionalAmount

    def get_dateflow(self):
        if self.legacy:
            future_dates, cash_flows, time_in_years = self._dateflow_og()
            return future_dates, cash_flows, time_in_years
        self.irregular_cpn = False
        self.get_date_range()
        self.irregular_cpn_check()
        self.get_cash_flows()
        self.get_years()
        #print [dt.strftime('%Y-%m-%d') for dt in self.future_dates]
        #print self.cash_flows
        return self.future_dates, pd.Series(self.cash_flows), pd.Series(self.time_in_years)

    def get_date_range(self):
        self.future_dates = [self.valuation]
        if self.first_cpn_dt is not None and type(self.first_cpn_dt) != pd.tslib.NaTType and self.first_cpn_dt != self.maturity:
            months = (self.maturity.year - self.first_cpn_dt.year)*12 + self.maturity.month - self.first_cpn_dt.month
            time_flow = pd.DatetimeIndex([self.maturity - DateOffset(months=e) for e in range(0, months+1, self.period)][::-1])
        else: #if first_cpn_dt is not specified we'll use the issue date to get the cashflow profile
            #print 'using issue date for date_flow. First coupon rate = ', self.first_cpn_dt
            months = (self.maturity.year - self.issue.year)*12 + self.maturity.month - self.issue.month
            #Issue1: Issue Date in same month as Maturity Date.
            #Impact: miss first cashflow
            #Conditions:    a) Issue Date and Maturity Date have the same month
            #               b) Maturity Date is not the last day in the month
            #               c) Issue Date less than Maturity Date
            #Action: if these conditions are met, add an extra cashflow
            if (self.issue.month == self.maturity.month) and (self.issue < self.maturity):
                months += 1
            time_flow = pd.DatetimeIndex([self.maturity - DateOffset(months=e) for e in range(0, months, self.period)][::-1])

        time_flow = [dt.to_pydatetime() for dt in time_flow]
        [self.future_dates.append(dt) for dt in time_flow if dt > self.valuation]

    def irregular_cpn_check(self):
        if self.last_cpn_dt is not None and type(self.last_cpn_dt) != pd.tslib.NaTType:
            if self.last_cpn_dt != self.future_dates[-2]:
                if self.last_cpn_dt.month == self.future_dates[-2].month:
                    return
                end = self.maturity
                start = self.valuation
                months = (end.year - start.year)*12 + end.month - start.month
                self.future_dates = pd.DatetimeIndex([end - DateOffset(months=e) for e in range(0, months, self.period)][::-1]).insert(0, start)
                self.future_dates = [dt.to_pydatetime() for dt in self.future_dates]
                if len(self.future_dates) == 1 and self.maturity != self.future_dates[-1]:
                    self.future_dates.append(self.maturity)
                self.irregular_cpn=True

    def get_cash_flows(self):
        self.coupon_rate = self.coupon_rate * self.notionalAmount
        n = len(self.future_dates)
        self.cash_flows = [0.0]
        for i in range(n)[1:]:
            if i == n - 1:
                self.cash_flows.append(self.notionalAmount + self.coupon_rate)
            else:
                self.cash_flows.append(self.coupon_rate)


    def get_years(self):
        self.time_in_years = [0.0]
        for i in range(len(self.future_dates))[1:]:
            # time_in_years.append(_actual_actual_daycount(self.future_dates[0], self.future_dates[i]))
            self.time_in_years.append(self._360_360_daycount(self.future_dates[0], self.future_dates[i]))

    def _actual_actual_daycount(self, start_dt, end_dt):
        '''
        Actual/Actual daycount method
        '''
        diffyears = end_dt.year - start_dt.year
        tmp_dt = start_dt + relativedelta(years=diffyears)
        difference = end_dt - tmp_dt
        # days_in_year = calendar.isleap(end_dt.year) and 366 or 365
        days_in_year = 365.25
        return diffyears + (difference.days + difference.seconds / 86400.0) / days_in_year

    def _360_360_daycount(self, day1, day2):
        '''
        360/360 daycount method
        '''
        if day2 > day1:
            number_of_days = 360.0 * (day2.year - day1.year) + 30.0 * \
                (day2.month - day1.month) + (day2.day - day1.day)
            # number_of_days = number_of_days % 360
            return number_of_days / 360.0
        else:
            return 0

    def coupon_payment_step(self, freq):
        if freq == 1.0:
            step = '12M'
        elif freq == 2.0:
            step = '6M'
        elif freq == 4.0:
            step = '3M'
        elif freq == 12.0:
            step = '1M'
        else:
            step = '6M'
        return step

    def coupon_payment_period(self, freq):
        if freq == 1.0:
            period = 12
        elif freq == 2.0:
            period = 6
        elif freq == 4.0:
            period = 3
        elif freq == 12.0:
            period = 1
        else:
            period = 6
        return period

    def _dateflow_og(self):
        import finance
        dateflow = finance.dateflow_generator(self.coupon_rate,
                                              enddate_or_integer=self.maturity,
                                              start_date=self.issue,
                                              step='6m',
                                              cashflowtype='bullit',
                                              profile='payment')

        valuationDateToTime = finance.DateToTime(
            valuation_date=self.valuation, daycount_method='360/360')

        timeFlow = finance.TimeFlow(
            date_to_time=valuationDateToTime, dateflow=dateflow)
        future_dates = timeFlow.future_dateflow().keys()
        cash_flows = timeFlow.discounted_values()
        time_in_years = timeFlow.future_times()
        return future_dates, cash_flows, time_in_years


    def _compare_og(coupon_rate, issue, maturity, valuation, future_dates):
        future_dates_og, cash_flows_og, time_in_years_og = _dateflow_og(coupon_rate, issue, maturity, valuation)
        for s in future_dates_og:
            if datetime.datetime.strptime(str(s), '%Y-%m-%d') not in future_dates:
                print 'coupon', coupon_rate
                print 'issue', issue
                print 'maturity', maturity
                print 'valuation', valuation
                print 'original'
                print future_dates_og, len(future_dates_og)
                print 'new'
                print [dt.strftime('%Y-%m-%d') for dt in future_dates], len(future_dates)
                break
        print cash_flows_og


    def _return_og(coupon_rate, issue, maturity, valuation):       
        future_dates_og, cash_flows_og, time_in_years_og = _dateflow_og(coupon_rate, issue, maturity, valuation)
        future_dates = [datetime.datetime.strptime(str(s), '%Y-%m-%d') for s in future_dates_og]
        cash_flows = [float(x) for x in cash_flows_og]
        time_in_years = [float(x) for x in time_in_years_og]

        return future_dates, cash_flows, time_in_years
