# Function to imply sample size estimation to Compare 2 Proportions: 
# 2-Sample Non-Inferiority or Superiority

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt
from typing import Tuple



def est_sample_size(pA: float, 
                    pB: float, 
                    k: float, 
                    delta: float,
                    alpha: float = 0.05, 
                    beta: float = 0.2,)-> Tuple[float, float]:
        
        """k: matching ratio
                    delta: testing margin
                    pA: proportion of A
                    pB: proportion of B
                    alpha: Type I error
                    beta: Type II error
                    """

        nB=(pA*(1-pA)/k + pB*(1-pB))*\
        ((norm.ppf(1-alpha)+norm.ppf(1-beta))/(pA-pB-delta))**2
                    
        nA = k*nB

        return nA, nB

## Simulation plot

def plot_sim(pA: float, 
                    pB: float, 
                    k: float, 
                    delta: float,
                    alpha: float = 0.05, 
                    beta: float = 0.2,
                    marg: float = 0.2):
    
    s_pA = np.linspace(max(0.01,pA-marg),min(1.,pA+marg),num = 50)
    s_pB = np.linspace(max(0.01,pB-marg),min(1.,pB+marg),num = 50)
    
    sd = np.linspace(min(-marg,delta - marg), max(delta + marg, 0.0), num = 50)
    sk = np.linspace(0.3, 1.5, num=50)
    
    plt.rc('font', size=12) 
    fig, axs = plt.subplots(2,2, figsize=(12,12))
    
    fig.suptitle('Kết quả mô phỏng')
    
    axs[0, 0].set_ylabel('Cỡ mẫu nhóm A')
    axs[0, 0].set_xlabel('Trung bình nhóm A')
    axs[0, 1].set_ylabel('Cỡ mẫu nhóm B')
    axs[0, 1].set_xlabel('Trung bình nhóm B')
    axs[1, 0].set_ylabel('Cỡ mẫu A+B')
    axs[1, 0].set_xlabel('Khoảng sai biệt delta')
    axs[1, 1].set_ylabel('Cỡ mẫu A+B')
    axs[1, 1].set_xlabel('Tỉ lệ nA/nB')
    
    nA, nB, tot = est_sample_size(pA = pA, pB = pB, k=k, \
                                 delta = delta, alpha = alpha, \
                                 beta = beta)
    
    for b in [0.05,0.1,0.2,0.25,0.3,]:
        
        sim_1 = [est_sample_size(pA = x, pB = pB, k=k, \
                                 delta = delta, alpha = alpha, \
                                 beta = b)[0] \
                 for x in s_pA]
        
        axs[0, 0].plot(s_pA, sim_1, label = f"power: {1-b}")
        
        sim_2 = [est_sample_size(pA = pA, pB = x, k=k, \
                                 delta = delta, alpha = alpha, \
                                 beta = b)[1] \
                 for x in s_pB]
        
        axs[0, 1].plot(s_pB, sim_2, label = f"power: {1-b}")
        
        sim_3 = [est_sample_size(pA = pA, pB = pB, k=k, \
                                 delta = x, alpha = alpha, \
                                 beta = b)[2] \
                 for x in sd]
        
        axs[1, 0].plot(sd, sim_3, label = f"power: {1-b}")
        
        sim_4 = [est_sample_size(pA = pA, pB = pB, k=x, \
                                 delta = delta, alpha = alpha, \
                                 beta = b)[2] \
                 for x in sk]
        
        axs[1, 1].plot(sk, sim_4, label = f"power: {1-b}")
        
        axs[0, 0].legend()
    
    plt.tight_layout()
    plt.show()