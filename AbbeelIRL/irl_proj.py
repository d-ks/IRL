# -*- coding: utf-8 -*-
"""
"Projection Method": Apprenticeship Learning via IRL

@author: D. Kishikawa
"""

import math
import numpy as np
from vi import ValueIteration

#二次元なのでγは配列
gamma = 0.9

x_size = 5
y_size = 5
max_S = 100
M_num = 5

#πE……4種類
pi_expert = np.array([[ 0,  1,  2,  3,  4,  9, 14, 19, 24],
                      [ 0,  5, 10, 15, 20, 21, 22, 23, 24],
                      [ 0,  1,  6,  7, 12, 13, 18, 19, 24],
                      [ 0,  5,  6, 11, 12, 17, 18, 23, 24]])


### Φ(s)の計算 ######################################
def phi(state):
    #one-hotベクトル化する
    phi_s = np.zeros(x_size*y_size)
    for i in range(x_size*y_size):
        if i == state:
            phi_s[i] = 1
        else:
            phi_s[i] = 0 
    #行列で返す
    return phi_s 

def Mu(policy):
    print("π: {}".format(policy))
    Mu_s = np.zeros(x_size*y_size)
    for s in range(policy.shape[0]):
        gamma_s = math.pow(gamma, s)
        Mu_s = Mu_s + gamma_s * phi(policy[s])        
    print("μ:")
    print(Mu_s)
    print("")
    return Mu_s

def MuE(policyE):
    MuE_m = np.zeros(x_size*y_size)

    
    for m in range(policyE.shape[0]):
        policy_Ex = np.zeros(x_size*y_size)
        MuE_s = np.zeros(x_size*y_size)
        
        policy_Ex = policyE[m,:]
        MuE_s = Mu(policy_Ex)
        
        MuE_m = MuE_m + MuE_s
        

    
    MuE_a = MuE_m / policyE.shape[0]
    
    print("-*- calculated μE -*- ")
    print(MuE_a)
    print("")
    
    return MuE_a

#################################################################################


Pi0 = np.array([0,5,0,5, 6,7,12,13,18,19,24])
mu0 = Mu(Pi0)
muE = MuE(pi_expert)


mu_i_1 = np.zeros(x_size*y_size)
mu_i_2 = np.zeros(x_size*y_size)

i = 1
w = np.zeros(x_size*y_size)
t = 0
epsilon = 0.005

while(True): ##########################################################
    
    print("-----------------iter: {}-------------------\n".format(i))
    
    # μ_ の計算
    if (i-1) == 0:
        mu_i_1 = mu0
    else:
        #Projection Methodの計算式
        N1 = np.dot( (mu_old - mu_i_2).T , (muE - mu_i_2) )
        N2 = np.dot( (mu_old - mu_i_2).T , (mu_old - mu_i_2) )
        N3 = N1 / N2
        N4 = np.dot(N3, (mu_old - mu_i_2))
        mu_i_1 = mu_i_2 + N4
    
    print("===== μ_ = {} =====\n".format(mu_i_1))
    
    # w(weight)の計算
    w = muE - mu_i_1
    print("===== w = {} =====\n".format(w))
    
    # tの計算・終了判定
    t = np.linalg.norm(muE - mu_i_1,ord=2)
    
    print("===== t = {} =====\n".format(t))
    
    if t <= epsilon :
        break
    
    R = np.zeros(x_size*y_size)
    
    #Rの計算
    for s in range(x_size*y_size):
        R[s] = np.dot(w, phi(s))
    
    print("===== R: =====")
    print(R)
    print("")

    
    #強化学習により、Rからpi_selectedを求める
    pi_selected = ValueIteration(R,5,5)
    #mu(pi_selected) を計算
    mu_old = Mu(pi_selected)
    mu_i_2 = mu_i_1
    #i を加算、ループ続行
    i = i + 1
    
##########################################################################
np.savetxt("R_irl.csv",R.reshape((x_size,y_size)), delimiter=",")