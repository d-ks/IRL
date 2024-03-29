# -*- coding: utf-8 -*-

"""
Maximum Entropy Inverse Reinforcement Learning + Adam optimizer
Python3 experimental implementation.
Developped : Tue June 29 2021
@author: D. Kishikawa
"""

import numpy as np
from makeP import makeProb
from tqdm import tqdm


def f(state, n_states):
    # return one-hot vector; 2 with n_states=4 -> [0 0 1 0]
    f_vec = np.zeros(n_states)
    f_vec[int(state)] = 1
    return f_vec


def MaxEntIRL(expert_traj, P, n_epoch=10000):
    ### Input:
    # expert_traj: numpy array, ( length_of_trajectories X num_of_trajectories );
    ## like [[0,1,2,2],[0,1,3,2], [0,1,2,3]]
    # P : numpy array, (S x A x S); state transition probability
    # n_epoch : number of learning iterations.

    # Adam param
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    m_t = 0
    v_t = 0

    # Get length of trajectory
    n_data, N = np.shape(expert_traj)
    n_states, n_actions, _ = np.shape(P)

    # Make initial state distribution
    P_init = np.zeros(n_states)
    for i in range(n_data):
        P_init[int(expert_traj[i][0])] += 1
    P_init /= np.sum(P_init)

    # Initialize variables
    R = np.zeros(n_states)
    Pas = np.zeros((n_states, n_actions))
    Dt = np.zeros((n_states, N))

    ################ Maximum Entropy Inverse Reinforcement Learning ##################################

    # Initialize theta
    theta = np.random.uniform(-0.01, 0.01, n_states)

    # Calculate f from expert
    f_exp = np.zeros(n_states)
    for i in range(n_data):
        traj = expert_traj[i, :]
        for s in range(N):
            f_exp += f(traj[s], n_states)
    f_exp /= n_data


    for i in tqdm(range(n_epoch)):

        # Compute current reward according to theta
        for s in range(n_states):
            R[s] = np.dot(theta, f(s, n_states))

        ##### Expected Edge Frequency Calculation #############

        ### Backward pass
        # 1.
        Zs = np.ones(n_states)
        # 2.
        PR = np.einsum("ijk,i->ijk", P, np.exp(R))  # This is constant in 1 iteration
        for _ in range(N):
            Za = np.einsum("ijk,k->ij", PR, Zs)
            Zs = np.sum(Za, axis=1)

        ### Local action probability computation
        # 3.
        for s in range(n_states):
            Pas[s, :] = Za[s, :] / Zs[s]

        ### Forward pass
        # 4.
        Dt[:, 0] = P_init
        # 5.
        for t in range(1, N):
            Dt[:, t] = np.einsum("i,ij,ijk->k", Dt[:, t - 1], Pas, P)

        ### Summing frequencies
        # 6.
        D = np.sum(Dt, axis=1)

        #######################################################

        # Compute gradient of theta according to Eq. (6)
        Df = np.zeros(n_states)
        for s in range(n_states):
            Df += D[s] * f(s, n_states)
        th_grad = f_exp - Df

        # Update theta with Gradient Ascent
        ## Adam implementation
        t = i + 1
        m_t = beta1 * m_t + (1 - beta1) * th_grad
        v_t = beta2 * v_t + (1 - beta2) * (th_grad**2)
        m_hat_t = m_t / (1 - (beta1 ** t))
        v_hat_t = v_t / (1 - (beta2 ** t))

        theta += alpha * (m_hat_t/(np.sqrt(v_hat_t) + epsilon))

    return R


#### test ####

x_size = 5
y_size = 5

expert_traj = np.array([[10, 11, 12, 13, 14, 9, 4]])

#####################################

n_states = int(x_size * y_size)

# make P of gridworld
P = makeProb(x_size, y_size)

# run maxent irl
R = MaxEntIRL(expert_traj, P)

print(R)

