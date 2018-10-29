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

    def __init__(self, datareader, urm, track_similarity):
        """
        Initialize the booster.
        :param datareader: a Datareader object
        :param track_similarity: a track-track similarity matrix in CSR format
        """
        self.datareader = datareader
        self.urm = urm
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

                    data.append(weighted_boost_value * gamma)
                    rows.append(p)
                    cols.append(index)

                # Increase position at each iteration
                pos += 1

        urm_boosted = sp.csr_matrix((data, (rows, cols)), shape=self.urm.shape)

        self.urm = pre.norm_l1_row(urm)
        urm_boosted = pre.norm_l1_row(urm_boosted)

        return self.urm + urm_boosted


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

    for a in [0.001]:
        for b in [1, 2, 3]:
            tb = TailBoost(dr, r_cfib, s)
            boosted = tb.boost(dr.target_playlists, last_tracks=b, gamma=a, k=1)
            score = ev.evaluation(boosted, urm, dr, save=False, name='best_cf_ib')
            print('%.5f' % (score))
