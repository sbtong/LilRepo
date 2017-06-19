import numpy as np
import scipy.optimize as opt


def compute_irr(time_in_years, cashflows, market_price=0.0, max_iter=100):

    def npv(irr):
        return compute_npv(time_in_years, cashflows, irr, market_price)

    def first_derivative(irr):
        return compute_analytic_first_derivative(time_in_years, cashflows, irr)

    initial_value = 0.0
    rate_of_return = opt.newton(npv, initial_value, fprime=first_derivative, maxiter=max_iter)
    return rate_of_return


def discount_cashflows_at_irr(time_in_years, cashflows, irr):
    return cashflows * np.exp(-irr * time_in_years)


def compute_npv(time_in_years, cashflows, irr, initial_value=0.0):
    return discount_cashflows_at_irr(time_in_years, cashflows, irr).sum() - initial_value


def compute_analytic_first_derivative(time_in_years, cashflows, irr):
    return np.sum(-time_in_years*discount_cashflows_at_irr(time_in_years, cashflows, irr))

def compute_macaulay_duration(time_in_years, cashflows, irr):
    npv = compute_npv(time_in_years, cashflows, irr)
    if npv == 0.0:
        return 0.0
    else:
        return np.sum(time_in_years*discount_cashflows_at_irr(time_in_years, cashflows, irr))/npv



