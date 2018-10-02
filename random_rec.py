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
        return "RandomRec"

    def fit(self):
        self.random_tracks = list(np.random.choice(20634, 50))
        self.random_tracks_str = ' '.join([str(i) for i in self.random_tracks])
        self.prediction = []

    def recommend(self, remove_seed=True):
        """
        Compute a single recommendation for a target playlist.
        :param target_playlist: the pid of the target playlist
        :param remove_seed: removed seed tracks
        :return: recommended_tracks or recommended_tracks_uri
        """

        urm = self.datareader.get_urm()

        # Recommend random tracks
        for t in self.datareader.target_playlists:

            if remove_seed:
                hold_ix = ~np.in1d(self.random_tracks, urm.indices[urm.indptr[t]:urm.indptr[t+1]])
                recommended_tracks = self.random_tracks[hold_ix]
                recommended_tracks = recommended_tracks[0:10]
                self.prediction.append([t, recommended_tracks])

            else:
                self.prediction.append([t, self.random_tracks_str])

        # Save CSV
        df = pd.DataFrame(self.prediction, columns=['playlist_id', 'track_ids'])
        df.to_csv(str(self) + '.csv', sep=',', index=False)


from datareader import Datareader

dr = Datareader()

rec = RandomRecommender(dr)
rec.fit()
rec.recommend()
