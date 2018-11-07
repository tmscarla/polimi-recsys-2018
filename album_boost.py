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
        self.urm = self.datareader.get_urm()
        self.eurm = pre.norm_l1_row(eurm)
        self.popularity = self.urm.sum(axis=0)
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

    def boost(self, target_playlists, last_tracks=2, k=5, gamma=0.1):
        """
        Boost the eurm for playlists with tracks given in order
        :param: k: the first top k tracks of the artist will be boosted
        :param last_tracks: number of last tracks to be considered
        :param: gamma: the weight of the boost
        :return: eurm: the boosted eurm
        """

        data = []
        rows = []
        cols = []

        df = self.datareader.train_df

        for p in tqdm(target_playlists[:5000], desc='AlbumBoost'):

            known_tracks = df.loc[df['playlist_id'] == p]['track_id'].values[::-1]

            lasts = []
            for j in range(last_tracks):
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
    dr = Datareader(train_new=2)
    ev = Evaluator()
    urm = dr.get_urm()
    eurm = dr.get_eurm_copenaghen()
    ab = AlbumBoost(dr, eurm=eurm)
    best = 0

    for lt in [1, 2, 3, 4]:
        for k in [2, 3, 4, 5, 6]:
            for g in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
                boosted = ab.boost(dr.target_playlists, last_tracks=lt, gamma=g, k=k)
                score = ev.evaluation(boosted, urm, dr, save=False, name='ab')
                if score > best: best = score

                print('gamma = ' + str(g))
                print('last = ' + str(lt))
                print('k = ' + str(k))
                print('score = %.5f' % score)
                print('best = %.5f' % best)
                print('-----------------------')

