import similaripy as sim
import scipy.sparse as sps
from evaluator import Evaluator
import numpy as np
import pandas as pd
from tqdm import tqdm


class CBFRecommender(object):
    """
    A random recommender. It recommends 10 random tracks for each playlist.
    """

    def __init__(self, datareader):
        self.datareader = datareader
        self.prediction = []

    def __str__(self):
        return "CBFRec"

    def fit(self, mode="cosine", al_id=True, ar_id=True, top_k=100):
        self.urm = self.datareader.get_urm()


        self.icm = self.datareader.get_icm(alid=al_id, arid=ar_id)

        # Train the model
        print("["+mode+"]")

        if mode == "cosine":
            self.model = sim.cosine(self.icm,
                                    k=top_k,
                                    verbose=True)
        elif mode == "as_cosine":
            self. model = sim.asymmetric_cosine(self.icm,
                                                alpha=0.7,
                                                k=top_k,
                                                verbose=True)
        elif mode == "dot":
            self. model = sim.dot_product(self.icm,
                                          k=top_k,
                                          verbose=True)





    def recommend(self, remove_seed=True):
        """
        Compute a single recommendation for a target playlist.
        :param remove_seed: removed seed tracks
        :param mode: #TODO
        :return: recommended_tracks or recommended_tracks_uri
        """

        # Compute user recommendations
        user_recommendations = sim.dot_product(self.urm,
                                               self.model,
                                               target_rows=list(self.datareader.target_playlists),
                                               k=100,
                                               verbose=False)

        # Recommend random tracks
        for t in self.datareader.target_playlists:

            scores = user_recommendations[t].toarray()[0]
            tracks = scores.argsort()[-100:][::-1]

            if remove_seed:
                hold_ix = ~np.in1d(tracks, self.urm.indices[self.urm.indptr[t]:self.urm.indptr[t+1]])
                recommended_tracks = tracks[hold_ix]
                recommended_tracks = recommended_tracks[0:10]
                recommended_tracks_str = ' '.join([str(i) for i in recommended_tracks])
                self.prediction.append([t, recommended_tracks_str])

            else:
                recommended_tracks_str = ' '.join([str(i) for i in tracks[:10]])
                self.prediction.append([t, recommended_tracks_str])

        # Save CSV
        df = pd.DataFrame(self.prediction, columns=['playlist_id', 'track_ids'])
        df.to_csv(str(self) + '.csv', sep=',', index=False)


if __name__ == '__main__':
    from datareader import Datareader
    dr = Datareader()

    rec = CBFRecommender(dr)
    rec.fit(mode="as_cosine", al_id=True, ar_id=True, top_k=50)
    rec.recommend()
    ev = Evaluator()

    prova_da_valutare = pd.read_csv(str(rec) + '.csv')
    dict_tua_sol = ev.csv_to_dict(prova_da_valutare)
    print(ev.evaluate_dict( dict_tua_sol ))




