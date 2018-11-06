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


class TailBoost(object):

    def __init__(self, datareader, eurm, track_similarity):
        """
        Initialize the booster.
        :param datareader: a Datareader object
        :param track_similarity: a track-track similarity matrix in CSR format
        """
        self.datareader = datareader
        self.eurm = eurm
        self.similarity = track_similarity

    def boost(self, target_playlists, last_tracks, k=5, gamma=0.1):
        """
        Boost the eurm for the playlists in specified categories..
        :param target_playlists: set of target playlists
        :param last_tracks: list of last tracks that will be boosted in each category
        :param: k: the first k simile tracks will be considered for boosting
        :param: gamma: the weight of the boost
        :return: urm_boosted: the boosted urm
        """

        data = []
        rows = []
        cols = []

        df = dr.train_df

        for p in tqdm(target_playlists[:5000], desc='TailBoost'):

            # Compute known tracks and invert them from last to first
            known_tracks = df.loc[df['playlist_id'] == p]['track_id'].values[::-1][:last_tracks]

            # Position
            pos = 0

            # Iterate for each track
            for track in known_tracks:
                # Slice row
                row_start = self.similarity.indptr[track]
                row_end = self.similarity.indptr[track + 1]

                row_columns = self.similarity.indices[row_start:row_end]
                row_data = self.similarity.data[row_start:row_end]

                # Compute top k simile tracks for track
                top_k = np.argsort(row_data, kind='mergesort')[::-1][:k]
                indices_to_boost = row_columns[top_k]
                boost_values = row_data[top_k]

                for i in range(len(indices_to_boost)):
                    index = indices_to_boost[i]

                    weighted_boost_value = boost_values[i] * log(pos + 1)

                    data.append(weighted_boost_value)
                    rows.append(p)
                    cols.append(index)

                # Increase position at each iteration
                pos += 1

        urm_boosted = sp.csr_matrix((data, (rows, cols)), shape=self.eurm.shape)

        self.eurm = pre.norm_l1_row(self.eurm)
        urm_boosted = pre.norm_l1_row(urm_boosted)

        return self.eurm + (urm_boosted * gamma)


if __name__ == '__main__':
    dr = Datareader(train_new=2)
    ev = Evaluator()
    urm = dr.get_urm()
    eurm_ens = dr.get_eurm_copenaghen()

    t_sim = sim.tversky(pre.bm25_row(urm.T), pre.bm25_col(urm), k=5000, alpha=0.30,
                             beta=0.50, verbose=False, format_output='csr')
    t_sim.data = np.power(t_sim.data, 0.75)

    for lt in [2, 4, 5, 6, 7, 8, 9]:
        for k in [2, 3, 5, 6, 7, 9]:
            for g in [0.0001, 0.0005, 0.001, 0.005, 0.01]:

                tb = TailBoost(dr, eurm_ens, t_sim)
                boosted = tb.boost(dr.target_playlists, last_tracks=lt, gamma=g, k=k)
                score = ev.evaluation(boosted, urm, dr, save=False, name='tb')

                print('gamma = ' + str(g))
                print('last = ' + str(lt))
                print('k = ' + str(k))
                print('%.5f' % score)
                print('-----------------------')
