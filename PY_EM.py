""" Example for EM algorithm. We throw two coins for 5 rounds and observe the following phenomenon.
[[1,-1,1,1,1],
[-1,-1,1,1,-1],
[1,-1,-1,-1,-1],
[1,-1,-1,1,1],
[1,1,1,1,-1]]
where 1 represent positive. -1 represents negative.
For each round, the coin is thrown 5 times. How to determine the probability of positive each coin is, and which coin is used in
 each round?"""

import numpy as np
from scipy import special as ssp


def em_song(observed, px_initial=0.4, py_initial=0.5):
    result_pos = np.sum((observed == 1).astype(float), axis=1)
    result_neg = np.sum((observed == -1).astype(float), axis=1)
    px, py = px_initial, py_initial
    px_old, py_old = 0, 0
    while px != px_old and py != py_old:
        # E-Step
        px_old = px
        py_old = py
        tmp1=ssp.comb(result_pos+result_neg,result_pos)*np.power(px,result_pos)*np.power(1-px,result_neg)
        tmp2=ssp.comb(result_pos+result_neg,result_pos)*np.power(py,result_pos)*np.power(1-py,result_neg)
        px=np.round(tmp1/(tmp1+tmp2)*100)/100
        py=1-px
        Ex_pos=result_pos*px
        Ex_neg=result_neg*px
        Ey_pos=result_pos*py
        Ey_neg=result_neg*py
        px=np.round(np.sum(Ex_pos)/np.sum((Ex_pos+Ex_neg))*100)/100
        py=np.round(np.sum(Ey_pos)/np.sum((Ey_neg+Ey_pos))*100)/100
    return px, py


observed_result = np.array([[1,1,1,1,1,-1,-1,-1,-1,-1],
                            [1,1,1,1,1,1,1,1,1,1],
                            [1,1,1,1,1,1,1,1,1,1],
                            [1,1,1,1,-1,-1,-1,-1,-1,-1],
                            [-1,-1,-1,1,1,1,1,1,1,1]])
d = em_song(observed_result, px_initial=0.3, py_initial=0.8)
print('Coin A refined prob. is %.2f,Coin B refined prob. is %.2f \n' % d[:2])
