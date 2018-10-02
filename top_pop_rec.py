import numpy as np
import pandas as pd
from tqdm import tqdm
from definitions import ROOT_DIR
from scipy import sparse as sps


class TopPopRecommender(object):
    """
    An unpersonalized top popular recommender. It recommends the top 500 tracks for each playlist.
    """

    def __init__(self, datareader):
        self.datareader = datareader


    def __str__(self):
        return "TopPop"

    def fit(self, top_k=50):
        """
        Train the recommender with a list of known interactions playlist - track
        :param top_k: k top tracks to be considered
        """

        self.prediction = []

        self.urm = self.datareader.get_urm()

        self.top_values = self.urm.sum(axis=0).A1
        self.top_tracks = self.top_values.argsort()[-top_k:][::-1]

    def recommend(self, remove_seed=True):
        """
        Compute a single recommendation for a target playlist.
        :param remove_seed: if true, tracks already present in the target_playlist are removed
        """

        # Recommend random tracks
        for t in self.datareader.target_playlists:

            if remove_seed:
                hold_ix = ~np.in1d(self.top_tracks, self.urm.indices[self.urm.indptr[t]:self.urm.indptr[t + 1]])
                recommended_tracks = self.top_tracks[hold_ix]
                recommended_tracks = recommended_tracks[0:10]
                recommended_tracks_str = ' '.join([str(i) for i in recommended_tracks])
                self.prediction.append([t, recommended_tracks_str])

            else:
                self.top_tracks_str = ' '.join([str(i) for i in self.top_tracks[0:10]])
                self.prediction.append([t, self.top_tracks_str])

        # Save CSV
        df = pd.DataFrame(self.prediction, columns=['playlist_id', 'track_ids'])
        df.to_csv(str(self) + '.csv', sep=',', index=False)


if __name__ == '__main__':
    from datareader import Datareader
    dr = Datareader()

    rec = TopPopRecommender(dr)
    rec.fit()
    rec.recommend()
