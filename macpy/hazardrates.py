import numpy as np
import pandas as pd



def SSpline(t,alpha,k):
    Spline=np.exp(-k*alpha*t)

    return Spline


def HazardUk(coupon, frequency, discountFactors,timeToCashflowInYears,RecoveryRate,alpha,DirtyPrice):
    Uk=[]
    #exponential splines up to degree 3
    Splinek=[1,2,3]
    #discountFactors and timetoCashflowInYears are assumed to be the same size:
    if len(discountFactors)!=len(timeToCashflowInYears):
        print "error: HazardUk: different sized arrays."

    for k in Splinek:
        Temp=0
        for i,t in enumerate(timeToCashflowInYears[1:-1]): #the first entry in timeToCashflowInYears is the valuation date so we don't want to include this, hence [1:
            i+=1
            Temp+= SSpline(t,alpha,k)*(
                coupon/frequency*discountFactors[i]-RecoveryRate*(discountFactors[i]-discountFactors[i+1])
        )
        Temp+=SSpline(timeToCashflowInYears.iloc[-1],alpha,k)*(
            (coupon/frequency+(1-RecoveryRate))*discountFactors[-1]
        )
        Uk.append(Temp)

    V = DirtyPrice/100.0 - RecoveryRate*discountFactors[1]  #first element of discountFactors is 0, corresponding to the discountFactor at the valuation date.

    return Uk, V


def CDSHazardBootstrap():

    HazardRates = [0.03]
    T=[5]

    return HazardRates,T

