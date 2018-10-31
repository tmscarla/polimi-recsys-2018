__author__ = "Tommaso Scarlatti"
__email__ = "scarlattitommaso@gmail.com"

from math import log
import numpy as np
import scipy.sparse as sp
import similaripy as sim
import implicit
import math
from tqdm import tqdm
from datareader import Datareader
from evaluator import Evaluator
from utils import pre_processing as pre
from album_boost import AlbumBoost
from collections import Counter
import xgboost as xgb


class XGBModel(object):

    def __init__(self, datareader):
        self.urm = datareader.get_urm()
        # self.train =


    def fit(self):
        pass


class XGBFeatureExtractor(object):

    def __init__(self, datareader):
        self.datareader = datareader
        self.urm = datareader.get_urm()
        self.urm_T = self.urm.T.tocsr()

        self.p_sim = sim.tversky(pre.bm25_row(urm), pre.bm25_col(urm.T), alpha=1, beta=1, k=70,
                                 shrink=0, target_rows=self.datareader.target_playlists, verbose=False)
        self.p_sim.data = np.power(s.data, 2.1)

        self.t_sim = sim.tversky(pre.bm25_row(urm.T), pre.bm25_col(urm), k=5000, alpha=0.30,
                                 beta=0.50, verbose=False, format_output='csr')
        self.t_sim.data = np.power(s.data, 0.75)

    def build_tracks_features(self):
        self.track_to_album = self.datareader.get_track_to_album_dict()
        self.track_to_artist = self.datareader.get_track_to_artist_dict()
        self.track_to_duration = self.datareader.get_track_to_duration()

    def build_playlist_features(self):
        n_playlists = self.urm.shape[0]

        self.p_pop, self.p_len, self.p_arh, self.p_alh,\
        self.p_artists, self.p_albums, self.p_dur, self.p_tracks = np.zeros(shape=n_playlists)

        tracks_pop = pre.norm_l1_row(self.urm.sum(axis=0))

        for p in range(n_playlists):
            p_tracks = self.urm.indices[self.urm.indptr[p]:self.urm.indptr[p+1]]
            p_artists = [self.track_to_artist[t] for t in p_tracks]
            p_albums = [self.track_to_album[t] for t in p_tracks]
            p_durations = [self.track_to_duration[t] for t in p_tracks]

            # Playlist's information
            self.p_tracks[p] = p_tracks
            self.p_artists[p] = p_artists
            self.p_albums[p] = p_albums

            # POPULARITY
            self.p_pop[p] = np.mean(tracks_pop[p_tracks])

            # LENGTH
            self.p_len[p] = len(p_tracks)

            # ARH
            self.p_arh[p] = math.log2(len(set(p_tracks)) / len(set(p_artists)))

            # ALH
            self.p_arh[p] = math.log2(len(set(p_tracks)) / len(set(p_albums)))

            # DURATION
            self.p_dur[p] = np.sum(p_durations)

    def build_models(self, verbose=True):
        # CF - IB
        # s = sim.tversky(pre.bm25_row(urm.T), pre.bm25_col(urm), k=5000, alpha=0.30, beta=0.50, verbose=verbose,
        #                 format_output='csr')
        # s.data = np.power(s.data, 0.75)
        self.r_cfib = pre.norm_l1_row(sim.dot_product(urm, self.t_sim.T, target_rows=t_ids, k=500, verbose=verbose))

        # CF - UB
        # s = sim.tversky(pre.bm25_row(urm), pre.bm25_col(urm.T), alpha=1, beta=1, k=70, shrink=0, target_rows=t_ids,
        #                 verbose=verbose)
        # s.data = np.power(s.data, 2.1)
        self.r_cfub = pre.norm_l1_row(sim.dot_product(self.p_sim, urm, k=500, verbose=verbose))

        # CB
        icm_al = dr.get_icm(alid=True, arid=False)
        icm_ar = dr.get_icm(alid=False, arid=True)
        icm = sp.hstack([icm_al * 1, icm_ar * 0.4])
        s = sim.dot_product(pre.bm25_col(icm), pre.bm25_col(icm.T), k=31, verbose=verbose, format_output='csr')
        s.data = np.power(s.data, 0.8)
        self.r_cb = pre.norm_l1_row(sim.dot_product(urm, s.T, target_rows=t_ids, k=500, verbose=verbose))

        # ALS
        model = implicit.als.AlternatingLeastSquares(factors=450, iterations=3, regularization=0.01)
        model.fit(urm.T)
        col, row, value = [], [], []
        for u in tqdm(t_ids):
            rec = model.recommend(userid=u, user_items=urm, N=150)
            for r in rec:
                row.append(u), col.append(r[0]), value.append(r[1])
        self.r_als = pre.norm_l1_row(sp.csr_matrix((value, (row, col)), shape=urm.shape))

        # ENSEMBLE
        r1 = pre.norm_l1_row(self.r_cf.tocsr())
        r2 = pre.norm_l1_row(self.r_cb.tocsr())
        self.r_ens = r1 + 0.04127 * r2

    def build_pairwise_feature(self, p, t):
        # ARTIST OVERLAPPING
        ar = self.track_to_artist[t]
        p_aro = Counter(self.p_artists[p])[ar] / self.p_artists

        # ALBUM OVERLAPPING
        al = self.track_to_album[t]
        p_alo = Counter(self.p_albums[p])[al] / self.p_albums

        # SIMILARITY t - seed tracks of p
        sim_t_seeds = np.average(self.t_sim.data[self.t_sim.indptr[t]:self.t_sim.indptr[t+1]][self.p_tracks])

        # SIMILARITY p - playlists that contains t
        p_with_t = self.urm_T.indices[self.urm_T.indptr[t]:self.urm_T.indptr[t+1]]
        sim_p_ps = np.average(self.p_sim.data[self.p_sim.indptr[p]:self.p_sim.indptr[p+1]][p_with_t])

        return p_aro, p_alo, sim_t_seeds, sim_p_ps

    def generate_samples(self, n_positive=None):

        for p in range(self.urm.shape[0]):
            p_tracks = self.urm.indices[self.urm.indptr[p]:self.urm.indptr[p+1]]

            if n_positive is None or n_positive > len(p_tracks):
                n_positive = len(p_tracks)

            # Extract seed tracks and an equal number of non-seed tracks randomly
            positives = np.random.choice(p_tracks, n_positive)
            negatives = []
            while len(negatives) < n_positive:
                neg = np.random.choice(self.urm.shape[1])
                if neg not in p_tracks:
                    negatives.append(neg)

            pos_and_neg = positives + negatives

            # Generate samples
            for i in range(len(pos_and_neg)):
                t = pos_and_neg[i]

                p_aro, p_alo, sim_t_seeds, sim_p_ps = self.build_pairwise_feature(p, t)








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



