import pandas as pd
import numpy as np
import scipy.sparse as sp
from definitions import ROOT_DIR
from utils import pre_processing as pre
import similaripy as sim


class Datareader(object):

    def __init__(self):
        self.playlists_df = pd.read_csv(ROOT_DIR + '/data/playlists.csv', sep=',')
        self.playlists = self.playlists_df['playlist_id'].values

        self.tracks_df = pd.read_csv(ROOT_DIR + '/data/tracks.csv', sep=',')
        self.tracks = self.tracks_df['track_id'].values

        self.train_df = pd.read_csv(ROOT_DIR + '/data/train.csv', sep=',')
        self.test_df = pd.read_csv(ROOT_DIR + '/data/local_test.csv', sep=',')

        self.target_playlists = pd.read_csv(ROOT_DIR + '/data/target_playlists.csv', sep=',')['playlist_id'].values

    def get_urm(self):
        rows = self.train_df['playlist_id'].values
        cols = self.train_df['track_id'].values

        urm = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                            shape=(len(self.playlists), len(self.tracks)),
                            dtype=np.int32)

        return urm

    def get_urm_test(self):
        rows = self.test_df['playlist_id'].values
        cols = self.test_df['track_id'].values

        urm_test = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                            shape=(len(self.playlists), len(self.tracks)),
                            dtype=np.int32)

        return urm_test


    def get_icm(self, alid=True, arid=True):
        assert alid or arid

        # Gather into lists
        rows = self.tracks_df['track_id'].values
        cols_alid = self.tracks_df['album_id'].values
        cols_arid = self.tracks_df['artist_id'].values

        # Album
        icm_alid = sp.csr_matrix((np.ones(len(rows)), (rows, cols_alid)),
                            	shape=(len(self.tracks), len(np.unique(cols_alid))),
                            	dtype=np.int32)

        # Artist
        icm_arid = sp.csr_matrix((np.ones(len(rows)), (rows, cols_arid)),
                                shape=(len(self.tracks), len(np.unique(cols_arid))),
                                dtype=np.int32)

        if alid and arid is False:
            return icm_alid
        elif arid and alid is False:
            return icm_arid
        else:
            icm = sp.hstack([icm_arid, icm_alid])
            return icm

    def get_eurm_copenaghen(self, verbose=False):
        urm = self.get_urm()
        t_ids = self.target_playlists

        # CF IB
        s = sim.tversky(pre.bm25_row(urm.T), pre.bm25_col(urm), k=5000, alpha=0.30, beta=0.50, verbose=verbose,
                        format_output='csr')
        s.data = np.power(s.data, 0.75)
        r_cfib = sim.dot_product(urm, s.T, target_rows=t_ids, k=500, verbose=verbose)

        # CF UB
        s = sim.tversky(pre.bm25_row(urm), pre.bm25_col(urm.T), alpha=1, beta=1, k=70, shrink=0, target_rows=t_ids,
                        verbose=verbose)
        s.data = np.power(s.data, 2.1)
        r_cfub = sim.dot_product(s, urm, k=500, verbose=verbose)

        # CF IB + UB
        r_cf = r_cfib + 3.15 * r_cfub

        # CB AL-AR
        icm_al = self.get_icm(alid=True, arid=False)
        icm_ar = self.get_icm(alid=False, arid=True)
        icm = sp.hstack([icm_al * 1, icm_ar * 0.4])
        s = sim.dot_product(pre.bm25_col(icm), pre.bm25_col(icm.T), k=31, verbose=verbose, format_output='csr')
        s.data = np.power(s.data, 0.8)
        r_cb = sim.dot_product(urm, s.T, target_rows=t_ids, k=500, verbose=verbose)

        # ENSEMBLE
        r1 = pre.norm_l1_row(r_cf.tocsr())
        r2 = pre.norm_l1_row(r_cb.tocsr())
        r_tot = r1 + 0.04127 * r2

        return pre.norm_l1_row(r_tot)

    def get_track_to_album_dict(self):
        """
        :return: dictionary: {track: album}
        """
        tracks_df = self.tracks_df

        keys = list(tracks_df['track_id'].values)
        values = list(tracks_df['album_id'].values)
        dictionary = dict(zip(keys, values))
        del tracks_df

        return dictionary

    def get_track_to_artist_dict(self):
        """
        :return: dictionary: {track: artist}
        """
        tracks_df = self.tracks_df

        keys = list(tracks_df['track_id'].values)
        values = list(tracks_df['artist_id'].values)
        dictionary = dict(zip(keys, values))
        del tracks_df

        return dictionary

    def get_track_to_duration(self):
        """
        :return: dictionary: {track: duration}
        """
        tracks_df = self.tracks_df

        keys = list(tracks_df['track_id'].values)
        values = list(tracks_df['duration_sec'].values)
        dictionary = dict(zip(keys, values))
        del tracks_df

        return dictionary
