"""
    the file contains all the default parameter settings
"""


root_folder = "C:\GitCodes\GuoRong\\"
max_random_test = 10
max_ga_iter = 1000
ga_pop_size = 50
#HeadwayFixed_H1 = 0.15
#HeadwayFlex_H2 = 0.25
invalidNum = -9999999
AreaLength_D = 10
AreaSide_s = 0.15
Time_lost = 0.0033
Time_pick = 0.0036
Passenger = -100
penalty = 1.0e4

WalkWeight_wa = 1
WaitWeight_ww = 1
TravelWeight_wt = 1

value_time = 20
value_dist = 2
value_hour = 20

FareFixed_f1 = 2
Fare_min = 2
Fare_max = 10

VehSpeed_v = 25
WalkSpeed_vw =2

HourMoney_cveh = 20
DistMoney_cdist = 2
TimeMoney_cvot = 20

StopFixed_N = AreaLength_D / AreaSide_s + 1

# weight_time = 1
# weight_profit = 1
nvars = 4
nbit_per_var = 16 
npop = 50  # population size 
ngen = 50000  # number of generations
 
lb = [0.0000001,Fare_min,0.0000001,0.0000001]
ub = [1,Fare_max,1,1]


def set_default_para():
    
    #HeadwayFixed_H1 = 0.15
    #HeadwayFlex_H2 = 0.25
    invalidNum = -9999999
    AreaLength_D = 10
    AreaSide_s = 0.15
    Time_lost = 0.0033
    Time_pick = 0.0036

    #Passenger = 500

    WalkWeight_wa = 1
    WaitWeight_ww = 1
    TravelWeight_wt = 1

    value_time = 20
    value_dist = 2
    value_hour = 20

    FareFixed_f1 = 2
    Fare_min = 2
    Fare_max = 8

    VehSpeed_v = 25
    WalkSpeed_vw =2

    HourMoney_cveh = 20
    DistMoney_cdist = 2
    TimeMoney_cvot = 20

    StopFixed_N = AreaLength_D / AreaSide_s + 1

    # weight_time = 1
    # weight_profit = 1
