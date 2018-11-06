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


class AlbumBoost(object):

    def __init__(self, datareader, eurm):
        """
        Initialize the booster.
        :param datareader: a Datareader object
        :param urm: the User Rating Matrix
        """
        self.datareader = datareader
        self.eurm = pre.norm_l1_row(eurm)
        self.popularity = urm.sum(axis=0)
        self.popularity = np.squeeze(np.asarray(self.popularity))
        self.track_to_album = self.get_track_to_album_dict()
        self.album_to_tracks = self.get_album_to_tracks_dict()

    def get_track_to_album_dict(self):
        """
        :return: dictionary: {track: album}
        """
        tracks_df = self.datareader.tracks_df

        keys = list(tracks_df['track_id'].values)
        values = list(tracks_df['album_id'].values)
        dictionary = dict(zip(keys, values))
        del tracks_df

        return dictionary

    def get_album_to_tracks_dict(self):
        """
        :return: dictionary: {album: [track1, track2, ...]}
        """
        tracks_df = self.datareader.tracks_df
        dictionary = dict()

        for index, row in tqdm(tracks_df.iterrows(), desc='Creating dictionary album to tracks'):

            alid = row['album_id']
            tid = row['track_id']

            if alid in dictionary.keys():
                dictionary[alid].append(tid)
            else:
                dictionary[alid] = [tid]

        return dictionary

    def boost(self, target_playlists, k=5, gamma=0.1):
        """
        Boost the eurm for playlists with tracks given in order
        :param: k: the first top k tracks of the artist will be boosted
        :param: gamma: the weight of the boost
        :return: eurm: the boosted eurm
        """

        data = []
        rows = []
        cols = []

        df = dr.train_df

        for p in tqdm(target_playlists[:5000], desc='AlbumBoost'):

            known_tracks = df.loc[df['playlist_id'] == p]['track_id'].values[::-1]

            lasts = []
            for j in range(2):
                lasts.append(self.track_to_album[known_tracks[j]])

            lasts = list(set(lasts))

            if len(lasts) == 1:
                tracks = np.array(self.album_to_tracks[lasts[0]])

                pop_indices = np.argsort(self.popularity[tracks])[::-1][:k]
                tracks_pop = tracks[pop_indices]

                for t in tracks_pop:
                    data.append(1)
                    rows.append(p)
                    cols.append(t)

        urm_boosted = sp.csr_matrix((data, (rows, cols)), shape=self.eurm.shape)

        return self.eurm + (urm_boosted * gamma)


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

