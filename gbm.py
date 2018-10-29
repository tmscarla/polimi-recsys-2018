__author__ = "Tommaso Scarlatti"
__email__ = "scarlattitommaso@gmail.com"

from math import log
import numpy as np
import scipy.sparse as sp
import similaripy as sim
from tqdm import tqdm
from datareader import Datareader
from evaluator import Evaluator
from utils import pre_processing as pre
from album_boost import AlbumBoost
import xgboost as xgb


class XGBModel(object):

    def __init__(self):
        self.urm =
        pass

    def fit(self):
        pass

    def feature_extraction(self):
        p_len =


if __name__ == '__main__':
    dr = Datareader()
    ev = Evaluator()
    urm = dr.get_urm()
    t_ids = dr.target_playlists
    verbose = False

    s = sim.tversky(pre.bm25_row(urm.T), pre.bm25_col(urm), k=5000, alpha=0.30, beta=0.50, verbose=verbose,
                    format_output='csr')

    s.data = np.power(s.data, 0.75)
    r_cfib = sim.dot_product(urm, s.T, target_rows=t_ids, k=500, verbose=verbose)
    score = ev.evaluation(r_cfib, urm, dr, save=False, name='best_cf_ib')
    print('%.5f' % (score))

    ab = AlbumBoost(dr, eurm=r_cfib, urm=urm)

    for g in [1]:
        for k in [10]:
            boosted = ab.boost(target_playlists=dr.target_playlists, k=k, gamma=g)
            score = ev.evaluation(boosted, urm, dr, save=True, name='copenhagen_boosted')
            print(g, k, '%.5f' % (score))



