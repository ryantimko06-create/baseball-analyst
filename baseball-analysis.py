
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # file render
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

SEED = 42
rng = np.random.default_rng(SEED)

def simulate_players(n=240): #simulator, arg n = # players

    #league average
    pa = rng.integers(250, 700, size=n) #plate appearances
    bb_rate = np.clip(rng.normal(0.085, 0.025, n), 0.02, 0.18) #base on ball rate
        #rng.normal for mean, std, #, --> array, low, high for np.clip
    hbp_rate = 0.01 #hbp 

    #simulation using binomial
    bb = rng.binomial(pa, bb_rate)
    hbp = rng.binomial(pa, hbp_rate)
    ab = pa - bb - hbp

    true_avg = np.clip(rng.normal(0.255, 0.028, n), 0.18, 0.34)
    hits = rng.binomial(ab, true_avg)

    #note -- np.clip returns array

    #hit distribution
    hr_share = np.clip(rng.normal(0.13, 0.06, n), 0.0, 0.15)
    d_share, t_share = 0.20, 0.02
    singles = np.zeros(n, dtype=int)
    doubles = np.zeros(n, dtype=int)
    triples = np.zeros(n, dtype=int)
    homers = np.zeros(n, dtype=int)

    for i in range(n):
        #singles prob, double prob, triple prob, hr prob
        probs = [1 - d_share - t_share - hr_share[i], d_share, t_share, hr_share[i]]

        #take total hits, probability, and give counts for each per player
        singles[i], doubles[i], triples[i], homers[i] = rng.multinomial(hits[i], probs)

    #pandas dataframe for easy scientic python graphing 
    return pd.DataFrame(
        {
            #cool syntax here I learned --> :03d to pad the 0s before a single digit number
            "player": [f"Player {i + 1:03d}" for i in range(n)],

            #return all previous rng arrays as dataframe
            "PA": pa,
            "AB": ab,
            "H": hits,
            "1B": singles,
            "2B": doubles,
            "3B": triples,
            "HR": homers,
            "BB": bb,
            "HBP": hbp,
        }
    )


