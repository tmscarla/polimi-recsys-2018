__author__ = "Tommaso Scarlatti"
__email__ = "scarlattitommaso@gmail.com"

from math import log
import time
import numpy as np
import pandas as pd
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
        self.datareader = datareader

    def fit(self, X_train, y_train, dump=False, verbose=False):
        if verbose:
            print('Start training xgb:')

        # Create a list of length of each group
        pids = list(dict.fromkeys(X_train['PID'].values))
        count = Counter(X_train['PID'].values)
        group = []

        for p in pids:
            group.append(count[p])

        # Remove TID and PID as features
        X_train = X_train.iloc[:, :-3]

        # Set group
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtrain.set_group(group)

        # Params and number of different trees to be built
        param = {
            'max_depth': 10,  # the maximum depth of each tree
            'eta': 0.3,  # the training step for each iteration
            'silent': 1,  # logging mode - quiet
            'objective': 'multi:softprob',
            'num_class': 2}  # the number of classes that exist in this datset
        num_round = 50  # the number of training iterations

        # Train model
        self.bst = xgb.train(param, dtrain, num_round, verbose_eval=True)

        if dump:
            self.bst.dump_model('dump.raw.txt')

    def predict(self, X_test):
        self.X_test = X_test.copy()

        pids = list(dict.fromkeys(X_test['PID'].values))
        count = Counter(X_test['PID'].values)
        group = []

        for p in pids:
            group.append(count[p])

        X_test.drop(['PID', 'TID'], axis=1, inplace=True)

        dtest = xgb.DMatrix(X_test)
        dtest.set_group(group)

        preds = self.bst.predict(dtest)

        return preds

    def softmax_preds_to_eurm(self, preds):
        rows = self.X_test['PID'].values
        cols = self.X_test['TID'].values

        data = preds[:, 1]

        eurm = sp.csr_matrix((data, (rows, cols)),
                             shape=(len(self.datareader.playlists), len(self.datareader.tracks)))

        return eurm


class XGBFeatureExtractor(object):

    def __init__(self, datareader):
        self.datareader = datareader
        self.urm = datareader.get_urm()
        self.urm_T = self.urm.T.tocsr()
        self.urm_test = datareader.get_urm_test()

        # Similarities
        self.p_sim = sim.tversky(pre.bm25_row(self.urm), pre.bm25_col(self.urm.T), alpha=1, beta=1, k=70,
                                 shrink=0, target_rows=self.datareader.target_playlists, verbose=False)
        self.p_sim.data = np.power(self.p_sim.data, 2.1)

        self.t_sim = sim.tversky(pre.bm25_row(self.urm.T), pre.bm25_col(self.urm), k=5000, alpha=0.30,
                                 beta=0.50, verbose=False, format_output='csr')
        self.t_sim.data = np.power(self.t_sim.data, 0.75)

        icm_al = self.datareader.get_icm(alid=True, arid=False)
        icm_ar = self.datareader.get_icm(alid=False, arid=True)
        icm = sp.hstack([icm_al * 1, icm_ar * 0.4])
        self.t_sim_cb = sim.dot_product(pre.bm25_col(icm), pre.bm25_col(icm.T), k=31, verbose=False, format_output='csr')
        self.t_sim_cb.data = np.power(self.t_sim_cb.data, 0.8)

        # Dataframes of samples
        self.cols = ['CFU', 'CFI', 'CB', 'ALS', 'P_LEN','P_DUR', 'ARH', 'ALH', 'P_POP', 'ARID',
                     'ALID', 'T_DUR', 'T_POP', 'T_TO_TP', 'P_TO_PT', 'ALO', 'ARO', 'PID', 'TID', 'TARGET']

        # Build singular features
        self.build_tracks_features()
        self.build_playlist_features()
        self.build_models()

    def select_cell(self, matrix, row, col):
        start = matrix.indptr[row]
        end = matrix.indptr[row+1]

        ix, = np.where(matrix.indices[start:end] == col)
        if any(ix):
            cell = matrix.data[start:end][ix]
            return cell[0]
        else:
            return 0.0

    def build_tracks_features(self):
        self.track_to_album = self.datareader.get_track_to_album_dict()
        self.track_to_artist = self.datareader.get_track_to_artist_dict()
        self.track_to_duration = self.datareader.get_track_to_duration()
        self.track_to_pop = pre.norm_l1_row(self.urm.sum(axis=0))[0]

    def build_playlist_features(self):
        n_playlists = self.urm.shape[0]

        self.p_pop = np.zeros(shape=n_playlists)
        self.p_len = np.zeros(shape=n_playlists)
        self.p_arh = np.zeros(shape=n_playlists)
        self.p_alh = np.zeros(shape=n_playlists)
        self.p_dur = np.zeros(shape=n_playlists)
        self.p_artists = [[]] * n_playlists
        self.p_albums = [[]] * n_playlists
        self.p_tracks = [[]] * n_playlists

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
            self.p_pop[p] = np.mean(self.track_to_pop[p_tracks])

            # LENGTH
            self.p_len[p] = len(p_tracks)

            # ARH
            self.p_arh[p] = math.log2(len(set(p_tracks)) / len(set(p_artists)))

            # ALH
            self.p_alh[p] = math.log2(len(set(p_tracks)) / len(set(p_albums)))

            # DURATION
            self.p_dur[p] = np.sum(p_durations)

    def build_models(self, verbose=True):
        # CF - IB
        self.r_cfib = pre.norm_l1_row(sim.dot_product(self.urm, self.t_sim.T,
                                                      target_rows=self.datareader.target_playlists,
                                                      k=500, verbose=verbose))

        # CF - UB
        self.r_cfub = pre.norm_l1_row(sim.dot_product(self.p_sim, self.urm, k=500, verbose=verbose))

        # CB
        self.r_cb = pre.norm_l1_row(sim.dot_product(self.urm, self.t_sim_cb.T,
                                                    target_rows=self.datareader.target_playlists,
                                                    k=500, verbose=verbose))

        # ALS
        model = implicit.als.AlternatingLeastSquares(factors=450, iterations=3, regularization=0.01)
        model.fit(self.urm_T)
        col, row, value = [], [], []
        for u in tqdm(self.datareader.target_playlists, desc='ALS'):
            rec = model.recommend(userid=u, user_items=self.urm, N=150)
            for r in rec:
                row.append(u), col.append(r[0]), value.append(r[1])
        self.r_als = pre.norm_l1_row(sp.csr_matrix((value, (row, col)), shape=self.urm.shape))

        # ENSEMBLE
        r_cf = self.r_cfib + 3.15 * self.r_cfub
        r1 = pre.norm_l1_row(r_cf.tocsr())
        r2 = pre.norm_l1_row(self.r_cb.tocsr())
        self.r_ens = r1 + 0.04127 * r2

    def build_pairwise_feature(self, p, t):
        # ARTIST OVERLAPPING
        ar = self.track_to_artist[t]
        p_aro = Counter(self.p_artists[p])[ar] / len(self.p_artists[p])

        # ALBUM OVERLAPPING
        al = self.track_to_album[t]
        p_alo = Counter(self.p_albums[p])[al] / len(self.p_albums)

        # SIMILARITY t - seed tracks of p
        start = self.t_sim.indptr[t]
        end = self.t_sim.indptr[t+1]
        ix = np.isin(self.t_sim.indices[start:end], self.p_tracks[p])

        if any(ix):
            vals = self.t_sim.data[start:end][ix]
            sim_t_seeds = np.mean(vals)

        else:
            sim_t_seeds = 0

        # SIMILARITY p - playlists that contains t
        p_with_t = self.urm_T.indices[self.urm_T.indptr[t]:self.urm_T.indptr[t+1]]

        start = self.p_sim.indptr[p]
        end = self.p_sim.indptr[p+1]
        ix = np.isin(self.p_sim.indices[start:end], p_with_t)

        if any(ix):
            vals = self.p_sim.data[start:end][ix]
            sim_p_ps = np.mean(vals)

        else:
            sim_p_ps = 0

        return p_aro, p_alo, sim_t_seeds, sim_p_ps

    def generate_training_samples(self, n_positive=None):
        df_list = []
        n_p = n_positive

        for p in tqdm(range(self.urm.shape[0]), desc='Generate training samples'):
            n_positive = n_p
            p_tracks = self.urm.indices[self.urm.indptr[p]:self.urm.indptr[p+1]]

            if n_positive is None or n_positive > len(p_tracks):
                n_positive = len(p_tracks)

            # Extract seed tracks and an equal number of non-seed tracks randomly
            positives = list(np.random.choice(p_tracks, n_positive))
            negatives = []
            while len(negatives) < n_positive:
                neg = np.random.choice(self.urm.shape[1])
                if neg not in p_tracks:
                    negatives.append(neg)

            pos_and_neg = positives + negatives

            # Generate samples
            for i in range(len(pos_and_neg)):
                # Select track
                t = pos_and_neg[i]

                # Check if is pos or neg and set target
                if i >= n_positive:
                    target = 0
                else:
                    target = 1

                # Build pairwaise features
                p_aro, p_alo, sim_t_seeds, sim_p_ps = self.build_pairwise_feature(p, t)

                df_list.append({'CFU': self.select_cell(self.r_cfub, p, t),
                                'CFI': self.select_cell(self.r_cfib, p, t),
                                'CB': self.select_cell(self.r_cb, p, t),
                                'ALS': self.select_cell(self.r_als, p, t),
                                'P_LEN': self.p_len[p],
                                'P_DUR': self.p_dur[p],
                                'ARH': self.p_arh[p],
                                'ALH': self.p_alh[p],
                                'P_POP': self.p_pop[p],
                                'ARID': self.track_to_artist[t],
                                'ALID': self.track_to_album[t],
                                'T_DUR': self.track_to_duration[t],
                                'T_POP': self.track_to_pop[t],
                                'T_TO_TP': sim_t_seeds,
                                'P_TO_PT': sim_p_ps,
                                'ALO': p_alo,
                                'ARO': p_aro,
                                'PID': p,
                                'TID': t,
                                'TARGET': target})

        self.df_train = pd.DataFrame(df_list)
        self.df_train = self.df_train[self.cols]
        self.df_train.to_csv('df_train.csv', index=False)

    def generate_test_samples(self, negative=False):
        df_list = []

        for p in tqdm(self.datareader.target_playlists, desc='Generate test samples'):
            p_tracks = self.urm_test.indices[self.urm_test.indptr[p]:self.urm_test.indptr[p + 1]]
            p_tracks = list(p_tracks)

            # Extract randomly negative tracks
            n_tracks = []
            while len(n_tracks) < len(p_tracks):
                neg = np.random.choice(self.urm.shape[1])
                if neg not in p_tracks:
                    n_tracks.append(neg)

            if negative:
                tracks = p_tracks + n_tracks
            else:
                tracks = p_tracks

            # Generate samples
            for i in range(len(tracks)):
                # Select track
                t = tracks[i]

                # Check if is pos or neg and set target
                if i >= len(p_tracks):
                    target = 0
                else:
                    target = 1

                # Build pairwaise features
                p_aro, p_alo, sim_t_seeds, sim_p_ps = self.build_pairwise_feature(p, t)

                df_list.append({'CFU': self.select_cell(self.r_cfub, p, t),
                                'CFI': self.select_cell(self.r_cfib, p, t),
                                'CB': self.select_cell(self.r_cb, p, t),
                                'ALS': self.select_cell(self.r_als, p, t),
                                'P_LEN': self.p_len[p],
                                'P_DUR': self.p_dur[p],
                                'ARH': self.p_arh[p],
                                'ALH': self.p_alh[p],
                                'P_POP': self.p_pop[p],
                                'ARID': self.track_to_artist[t],
                                'ALID': self.track_to_album[t],
                                'T_DUR': self.track_to_duration[t],
                                'T_POP': self.track_to_pop[t],
                                'T_TO_TP': sim_t_seeds,
                                'P_TO_PT': sim_p_ps,
                                'ALO': p_alo,
                                'ARO': p_aro,
                                'PID': p,
                                'TID': t,
                                'TARGET': target})

        self.df_test = pd.DataFrame(df_list)
        self.df_test = self.df_test[self.cols]
        self.df_test.to_csv('df_test.csv', index=False)

    def generate_prediction_samples(self, eurm, k):
        df_list = []

        for p in tqdm(self.datareader.target_playlists, desc='Generate prediction samples'):
            val = eurm.data[eurm.indptr[p]:eurm.indptr[p+1]]
            ind = val.argsort()[-k:][::-1]
            p_tracks = list(eurm[p].indices[ind])

            # Generate samples
            for i in range(len(p_tracks)):
                # Select track
                t = p_tracks[i]

                # Build pairwaise features
                p_aro, p_alo, sim_t_seeds, sim_p_ps = self.build_pairwise_feature(p, t)

                df_list.append({'CFU': self.select_cell(self.r_cfub, p, t),
                                'CFI': self.select_cell(self.r_cfib, p, t),
                                'CB': self.select_cell(self.r_cb, p, t),
                                'ALS': self.select_cell(self.r_als, p, t),
                                'P_LEN': self.p_len[p],
                                'P_DUR': self.p_dur[p],
                                'ARH': self.p_arh[p],
                                'ALH': self.p_alh[p],
                                'P_POP': self.p_pop[p],
                                'ARID': self.track_to_artist[t],
                                'ALID': self.track_to_album[t],
                                'T_DUR': self.track_to_duration[t],
                                'T_POP': self.track_to_pop[t],
                                'T_TO_TP': sim_t_seeds,
                                'P_TO_PT': sim_p_ps,
                                'ALO': p_alo,
                                'ARO': p_aro,
                                'PID': p,
                                'TID': t})

        self.df_predict = pd.DataFrame(df_list)
        cols = self.cols.copy()
        cols.remove('TARGET')
        self.df_predict = self.df_predict[cols]
        self.df_predict.to_csv('df_predict_' + str(k) + '.csv', index=False)


if __name__ == '__main__':
    # dr = Datareader()
    # ev = Evaluator()
    # urm = dr.get_urm()
    #
    # pred_vector = pd.read_csv('preds_map_10.csv').values.flatten()
    # print(len(pred_vector))
    #
    # X_test = pd.read_csv('df_predict_10.csv')
    # tids = X_test['TID'].values
    # pids = X_test['PID'].values
    #
    # r = []
    # c = []
    # d = []
    #
    # for i in range(0, len(pred_vector), 10):
    #     p = pids[i]
    #     top = np.argsort(pred_vector[i:i+10])[::-1]
    #     tracks = tids[i:i+10][top]
    #
    #     for j in range(10):
    #         r.append(p)
    #         c.append(tracks[j])
    #         d.append(10-j)
    #
    # eurm_prova = sp.csr_matrix((d, (r, c)), shape=urm.shape)
    # score = ev.evaluation(eurm_prova, urm, dr, save=False, name='prova')
    # print(score)
    #
    # exit()

    dr = Datareader()
    ev = Evaluator()
    urm = dr.get_urm()
    eurm = dr.get_eurm_copenaghen()

    # Remove seeds
    urm_max = urm.copy()
    urm_max.data = np.ones(len(urm.data)) * eurm.max()
    eurm_no_seeds = eurm - urm_max

    # xgb_fe = XGBFeatureExtractor(dr)
    # xgb_fe.generate_training_samples()
    # xgb_fe.generate_test_samples()
    # xgb_fe.generate_prediction_samples(eurm_no_seeds, 30)

    train = pd.read_csv('df_train.csv')
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    X_test = pd.read_csv('df_predict_30.csv')

    xgbmodel = XGBModel(dr)
    xgbmodel.fit(X_train=train, y_train=y_train)
    preds = xgbmodel.predict(X_test)
    # preds_df = pd.DataFrame(preds)
    # preds_df.to_csv('preds_soft_10.csv', index=False)
    eurm_xgb = xgbmodel.softmax_preds_to_eurm(preds)
    print(eurm.shape)

    eurm_xgb = pre.norm_l1_row(eurm_xgb)

    for a in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        eurm_ens = (eurm * a) + ((1-a) * eurm_xgb)

        score = ev.evaluation(eurm_ens, urm, dr, save=True, name='xgb')
        print(a)
        print(score)
    exit()




