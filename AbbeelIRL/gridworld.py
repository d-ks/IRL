import numpy as np

def makeP(x_size, y_size, n_action=4):
    # make transition prob

    n_states = x_size*y_size
    # S, A, S'
    P = np.zeros(( n_states, n_action, n_states ))



    for s in range(n_states):

        #left is wall
        if s%x_size == 0:
            if s == 0:
                P[0][1][5] = 1
                P[0][3][1] = 1
            elif s > (x_size*(y_size-1)-1):
                P[s][0][s-x_size] = 1
                P[s][3][s+1] = 1
            else:
                P[s][0][s-x_size] = 1
                P[s][1][s+x_size] = 1
                P[s][3][s+1] = 1

        #right is wall
        elif (s+1)%x_size == 0:
            if s == (x_size*y_size-1):
                P[s][0][s-x_size] = 1
                P[s][2][s-1] = 1
            elif s < x_size:
                P[s][1][s+x_size] = 1
                P[s][2][s-1]=1
            else:
                P[s][0][s-x_size] = 1
                P[s][1][s+x_size] = 1
                P[s][2][s-1] = 1

        #upper is wall
        elif s < x_size and s%x_size!=0 and (s+1)%x_size !=0 :
            P[s][1][s+x_size] = 1
            P[s][2][s-1] = 1
            P[s][3][s+1] = 1

        #lower is wall
        elif s > (x_size*(y_size-1)-1) and s%x_size!=0 and (s+1)%x_size !=0:
            P[s][0][s-x_size] = 1
            P[s][2][s-1] = 1
            P[s][3][s+1] = 1

        else:
            P[s][0][s-x_size] = 1
            P[s][1][s+x_size] = 1
            P[s][2][s-1] = 1
            P[s][3][s+1] = 1

    return P