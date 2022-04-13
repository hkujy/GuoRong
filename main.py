# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 19:51:26 2022

@author: ronguo
"""

"""
遗传算法求最优解
目标函数：总的时间效益（运营商和乘客）+ 运营商收益

变量：
x[0]:beita
x[1]:FareFlex_f2
x[2]:HeadwayFixed_H1
x[3]:HeadwayFlex_H2
"""
from cmath import log
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import pandas as pd
import matplotlib.pyplot as plt
import default_para as dp


def f(x):
    #Notes: you need to define before using 
    FixedArea_a1 = dp.invalidNum
    WalkDist_l1 = dp.invalidNum
    WalkDist_l2 = dp.invalidNum
    #--------------------------------
    if x[0] > 0 and x[0] <= 0.5:
        FixedArea_a1 = 2 * (x[0] ** 2)
        WalkDist_l1 = 8 * dp.AreaSide_s * x[0] / 3
        WalkDist_l2 = 4 * dp.AreaSide_s * x[0] / 3
    if x[0]>0.5 and x[0]<=1:
        FixedArea_a1 = 1 - 2 * (1 - x[0]) ** 2
        WalkDist_l1 = (6 * dp.AreaSide_s - 8 * (1 - x[0]) ** 2 * (2 * dp.AreaSide_s * x[0] + dp.AreaSide_s)) / (3 - 6 * (1 - x[0]) ** 2)
        WalkDist_l2 = (3 * dp.AreaSide_s - 4 * (1 - x[0]) ** 2 * (2 * dp.AreaSide_s * x[0] + dp.AreaSide_s)) / (3 - 6 * (1 - x[0]) ** 2)
    else:
        print("Debug: Undefined x[0]")

    FlexArea_a2 = 1 - FixedArea_a1
    WalkDist_l3 = WalkDist_l2

    # 四类乘客比例
    PassengerType_p1 = FixedArea_a1 ** 2
    PassengerType_p2 = FixedArea_a1 * FlexArea_a2
    PassengerType_p3 = FlexArea_a2 * FixedArea_a1
    PassengerType_p4 = FlexArea_a2 ** 2


    #TODO: passengers here is not defined
    # 车辆行驶距离
    DistFixed_d1 = 2 * dp.AreaLength_D / x[2]
    DistFlex_d2 = 2 * dp.AreaLength_D / x[3] + 4 * dp.AreaLength_D * dp.AreaSide_s ** 2 * Passenger * FlexArea_a2 / 3

    # 车队数量
    #  之前
    # FleetsizeFixed_m1 = DistFixed_d1 / VehSpeed_v + 2 * (StopFixed_N - 1) * Time_lost / x[2]
    # FleetsizeFlex_m2 = DistFlex_d2 / VehSpeed_v + 4 * Time_pick * AreaLength_D * AreaSide_s * Passenger * FlexArea_a2
    #现在(输出)
    FleetsizeFixed_m1 = 1/x[2]
    FleetsizeFlex_m2 = 1/x[3]


    # 车辆运行成本
    OperCostFixed_c1 = HourMoney_cveh * FleetsizeFixed_m1 + DistMoney_cdist * DistFixed_d1
    OperCostflex_c2 = HourMoney_cveh * FleetsizeFlex_m2 + DistMoney_cdist * DistFlex_d2
    TotalCost_c = OperCostFixed_c1+OperCostflex_c2

    #为了计算乘车时间，先计算一些质心距离
    MassHMDNG_lq4 = (3 * AreaSide_s - 8 * AreaSide_s * x[0] ** 3) / (6 * (1 - 2 * x[0] ** 2))
    MassDist_l0 = (2 * x[0] * AreaSide_s + AreaSide_s) / 3

    #乘车距离
    TraDistFixed_r1 = 0.34 * AreaLength_D
    if x[0] > 0 and x[0] <= 0.5:
        TraDisFlex_r2 = 2 * FlexArea_a2 * (DistFlex_d2 * x[3] / (2 * AreaLength_D)) * MassHMDNG_lq4


    else:
        TraDisFlex_r2 = 2 * FlexArea_a2 * (DistFlex_d2 * x[3] / (2 * AreaLength_D)) * MassDist_l0



    # 乘客时间计算
    WalkTime_A = (WalkDist_l1 * PassengerType_p1 + WalkDist_l2 * PassengerType_p2 +
                  WalkDist_l3 * PassengerType_p3) / WalkSpeed_vw

    WaitTime_W = (x[2] / 2) * PassengerType_p1 + (x[2] + x[3]) * PassengerType_p2 / 2 + \
                 (x[2] / 2 + x[3]) * PassengerType_p4

    TravelTime_T = (TraDistFixed_r1 * PassengerType_p1 +
                    (TraDistFixed_r1+TraDisFlex_r2) * PassengerType_p2 * 2 +
                    (TraDistFixed_r1+TraDisFlex_r2*2) * PassengerType_p4) / VehSpeed_v
    TotalUserTime = (WalkWeight_wa * WalkTime_A + WaitWeight_ww * WaitTime_W + TravelWeight_wt * TravelTime_T)


    if 2 * Passenger * AreaLength_D * AreaSide_s * (FareFixed_f1 * (PassengerType_p1 + PassengerType_p2 * 2) +
                                                    x[1] * (PassengerType_p2 * 2 + PassengerType_p4)) - TotalCost_c < 0:
        x[1] = Fare_max
    else:
        pass

    AgencyCost_dist = value_dist / (Passenger * AreaLength_D * AreaSide_s * value_time)
    AgencyCost_hour = value_hour / (Passenger * AreaLength_D * AreaSide_s * value_time)
    TotalAgencyTime = AgencyCost_dist * (DistFixed_d1 + DistFlex_d2) + AgencyCost_hour * (FleetsizeFixed_m1 + FleetsizeFlex_m2)
    TotalFare = 2 * Passenger * AreaLength_D * AreaSide_s *(FareFixed_f1 * (PassengerType_p1 + PassengerType_p2 * 2) + x[1] * (PassengerType_p2 * 2 + PassengerType_p4 * 2))

    TotalTimeProfit = (TotalUserTime+TotalAgencyTime)*value_time
    AgencyProfit_pi = TotalFare-TotalCost_c

    #目标函数
    #目前
    Opt_function_Z = TotalTimeProfit + 1/AgencyProfit_pi
    # 考虑新修改
    # Opt_function_Z = TotalTimeProfit - AgencyProfit_pi


    res.append([Passenger, x[0], x[1], x[2], x[3],  Opt_function_Z,
                TotalTimeProfit, AgencyProfit_pi, TotalAgencyTime,TotalUserTime,TotalFare,TotalCost_c,
                DistFixed_d1, DistFlex_d2, FleetsizeFixed_m1, FleetsizeFlex_m2,
                WalkTime_A, WaitTime_W, TravelTime_T])


    return Opt_function_Z




varbound=np.array([[0,1], [Fare_min, Fare_max],[0,1],[0,1]])
vartype=np.array([['real'],['real'],['real'],['real']])

df = pd.DataFrame(columns=['Passenger','opt_beta','opt_basefare','HeadwayFixed_H1','HeadwayFlex_H2','Opt_function_Z',
                           'TotalTimeProfit', 'AgencyProfit_pi', 'TotalAgencyTime','TotalUserTime','TotalFare','TotalCost_c',
                           'DistFixed_d1','DistFlex_d2','FleetsizeFixed_m1','FleetsizeFlex_m2',
                           'WalkTime_A','WaitTime_W','TravelTime_T'])


algorithm_param = {'max_num_iteration': 100,\
                   'population_size':50,\
                   'mutation_probability':0.1,\
                   'elit_ratio': 0.01,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.3,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}

minPas_fare=[]
for i in range(5,501,55):  #乘客的数量
    Passenger = i
    minPas_fare.append(5)
    for j in range(0,5):
        res = []
        model = ga(function=f, dimension=4, variable_type_mixed=vartype, variable_boundaries=varbound,algorithm_parameters=algorithm_param)
        model.run()
        ans = res[-1]
        minPas_fare[-1]=min(minPas_fare[-1],ans[2])
        # print(ans)
        df = df.append([{'Passenger':ans[0],'opt_beta':ans[1],'opt_basefare':ans[2],'HeadwayFixed_H1':ans[3],'HeadwayFlex_H2':ans[4],'Opt_function_Z':ans[5],
                         'TotalTimeProfit':ans[6], 'AgencyProfit_pi':ans[7], 'TotalAgencyTime':ans[8], 'TotalUserTime':ans[9], 'TotalFare':ans[10],'TotalCost_c':ans[11],
                         'DistFixed_d1':ans[12],'DistFlex_d2':ans[13],'FleetsizeFixed_m1':ans[14],'FleetsizeFlex_m2':ans[15],
                         'WalkTime_A':ans[16],'WaitTime_W':ans[17],'TravelTime_T':ans[18]}])

def deal():
     df = pd.DataFrame(minPas_fare,columns = ['fare'])
     df.to_excel('minPas_fare.xlsx', encoding="utf-8", index=None)

if __name__ == "__main__":
    deal()

df_mean = df.groupby(by = 'Passenger').mean().reset_index()


print(minPas_fare)
plt.rcParams['font.sans-serif']=['Times New Roman']

print(df)
print(df_mean)
df.to_excel('result_total.xlsx', encoding="utf-8", index=None)
df_mean.to_excel('result_mean.xlsx',encoding="utf-8", index=None)

plt.rcParams['font.sans-serif']=['Times New Roman']
#df.head()


