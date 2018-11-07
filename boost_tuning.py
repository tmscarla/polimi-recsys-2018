from math import log
import numpy as np
import scipy.sparse as sp
import similaripy as sim
from tqdm import tqdm
from datareader import Datareader
from evaluator import Evaluator
from utils import pre_processing as pre
from tail_boost import TailBoost
from album_boost import AlbumBoost

if __name__ == '__main__':
    dr = Datareader(train_new=2)
    ev = Evaluator()
    urm = dr.get_urm()
    eurm_ens = dr.get_eurm_copenaghen()

    t_sim = sim.tversky(pre.bm25_row(urm.T), pre.bm25_col(urm), k=5000, alpha=0.30,
                             beta=0.50, verbose=False, format_output='csr')
    t_sim.data = np.power(t_sim.data, 0.75)

    # Album boost
    ab = AlbumBoost(dr, eurm=eurm_ens)
    eurm_ab = ab.boost(dr.target_playlists, last_tracks=2, k=6, gamma=0.01)

    # Tail boost
    tb = TailBoost(dr, eurm=eurm_ab, track_similarity=t_sim)
    eurm_final = tb.boost(dr.target_playlists, last_tracks=2, k=9, gamma=0.01)

    # Evaluation
    score = ev.evaluation(eurm_final, urm, dr, save=False, name='boosts')
    print(score)
