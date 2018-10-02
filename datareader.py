import pandas as pd
import numpy as np
import scipy.sparse as sp
from definitions import ROOT_DIR


class Datareader(object):

    def __init__(self):
        self.playlists_df = pd.read_csv(ROOT_DIR + '/data/playlists.csv', sep=',')
        self.playlists = self.playlists_df['playlist_id'].values

        self.tracks_df = pd.read_csv(ROOT_DIR + '/data/tracks.csv', sep=',')
        self.tracks = self.tracks_df['track_id'].values

        self.train_df = pd.read_csv(ROOT_DIR + '/data/train.csv', sep=',')

        self.target_playlists = pd.read_csv(ROOT_DIR + '/data/target_playlists.csv', sep=',')['playlist_id'].values

    def get_urm(self):
        rows = self.train_df['playlist_id'].values
        cols = self.train_df['track_id'].values

        urm = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                            shape=(len(self.playlists), len(self.tracks)),
                            dtype=np.int32)

        return urm

    def get_icm(self, alid=True, arid=True):
        assert alid or arid

        # Gather into lists
        rows = self.tracks_df['track_id'].values
        cols_alid = self.tracks_df['album_id'].values
        cols_arid = self.tracks_df['artist_id'].values

        # Album
        icm_alid = sp.csr_matrix((np.ones(len(rows)), (rows, cols_alid)),
                            shape=(len(self.tracks), len(cols_alid)),
                            dtype=np.int32)

        # Artist
        icm_arid = sp.csr_matrix((np.ones(len(rows)), (rows, cols_arid)),
                                 shape=(len(self.tracks), len(cols_arid)),
                                 dtype=np.int32)

        if alid and arid is False:
            return icm_alid
        elif arid and alid is False:
            return icm_arid
        else:
            icm = sp.hstack([icm_arid, icm_alid])
            return icm

