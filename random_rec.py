import numpy as np
import pandas as pd
from tqdm import tqdm

class RandomRecommender(object):
    """
    A random recommender. It recommends 10 random tracks for each playlist.
    """

    def __init__(self, datareader):
        self.datareader = datareader

    def __str__(self):
        return "Random Recommender"

    def fit(self):
        self.random_tracks = list(np.random.choice(20634, 10))
        self.prediction = []

    def recommend(self, remove_seen=True):
        """
        Compute a single recommendation for a target playlist.
        :param target_playlist: the pid of the target playlist
        :param remove_seen: if true, tracks already present in the target_playlist are removed
        :return: recommended_tracks or recommended_tracks_uri
        """

        for t in self.datareader.target_playlists:
            self.prediction.append([t, self.random_tracks])

        df = pd.DataFrame(self.prediction)
        df.to_csv('prova.csv', sep=',')


        # # Remove known tracks from the prediction
        # if remove_seen and target_playlist in seen.index:
        #     hold_ix = ~np.in1d(self.random_tracks, seen[target_playlist])
        #     recommended_tracks = self.random_tracks[hold_ix]
        #     recommended_tracks = recommended_tracks[0:500]
        # else:
        #     recommended_tracks = self.random_tracks[0:500]
        #
        # # Return tids or uris
        # if is_submission:
        #     recommended_tracks_uri = [self.tracks_df['track_uri'][t] for t in recommended_tracks]
        #     return recommended_tracks_uri
        # else:
        #     return recommended_tracks

from datareader import Datareader

dr = Datareader()

rec = RandomRecommender(dr)
rec.fit()
rec.recommend()
