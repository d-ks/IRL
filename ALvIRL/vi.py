# Value Iteration


import numpy as np
from gridworld import makeP


def ValueIteration(R, x, y):

    n_states = x * y

    V = np.random.uniform(0, 1, n_states)
    Q = np.zeros(4)

    P = makeP(x, y)

    theta = 0.1 ** 10
    delta = 100

    i = 0

    print("... start VI ...")

    while delta > theta:
        v = np.copy(V)
        for s in range(n_states):
            for a in range(4):
                Q[a] = R[s] + 0.95 * np.dot(P[s][a][:], V)
            V[s] = np.max(Q)
        delta = np.linalg.norm(v - V)
        i += 1

    print("... gen opt policy ...")
    goal = int(x * y - 1)

    s = 0
    traj = []

    maxlen = 100
    leng = 0
    while s != goal:
        traj.append(s)
        leng += 1
        if leng >= maxlen:
            break

        V_list = np.zeros(4)
        try:
            V_list[0] = V[int(s + 1)]
        except:
            V_list[0] = -np.inf
        try:
            if s - 1 > -1:
                V_list[1] = V[int(s - 1)]
            else:
                V_list[1] = -np.inf
        except:
            V_list[1] = -np.inf
        try:
            V_list[2] = V[s + x]
        except:
            V_list[2] = -np.inf
        try:
            if s - x > -1:
                V_list[3] = V[int(s - x)]
            else:
                V_list[3] = -np.inf
        except:
            V_list[3] = -np.inf
        maxIndex = [i for i, x in enumerate(V_list) if x == max(V_list)]
        na = np.random.choice(maxIndex)
        if na == 0:
            ns = s + 1
        elif na == 1:
            ns = s - 1
        elif na == 2:
            ns = s + x
        elif na == 3:
            ns = s - x
        s = int(ns)

    traj.append(int(x * y - 1))
    traj_numpy = np.array(traj, dtype=int)
    return traj_numpy

