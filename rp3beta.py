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

    def __init__(self, datareader, verbose=False):
        self.datareader = datareader
        self.prediction = []
        self.verbose = verbose

    def __str__(self):
        return "rp3beta"

    def fit(self, a=1, b=1, top_k=100):
        self.urm = self.datareader.get_urm()


        # Train the model

        self. model = sim.rp3beta(self.urm.T,
                                  alpha=a,
                                  beta=b,
                                  k=top_k,
                                  verbose=self.verbose)





    def recommend(self, remove_seed=True):
        """
        Compute a single recommendation for a target playlist.
        :param remove_seed: removed seed tracks
        :return: recommended_tracks or recommended_tracks_uri
        """

        # Compute user recommendations
        self.eurm = sim.dot_product(self.urm,
                                               self.model,
                                               target_rows=list(self.datareader.target_playlists),
                                               k=100,
                                               verbose=self.verbose)



if __name__ == '__main__':
    from datareader import Datareader
    dr = Datareader()

    # best = {"map":0, "alpha":0, "beta":0}
    # for a in np.arange(0.0, 1.0, 0.1):
    #     for b in np.arange(0.0, 1.0, 0.1):
    #
    #         rec = CBFRecommender(dr)
    #         rec.fit(top_k=50,a=a,b=b)
    #         rec.recommend()
    #         ev = Evaluator()
    #
    #         res = ev.evaluation(rec.eurm, rec.urm, dr)
    #         print("alpha:", a, "beta:", b, " map:", res)
    #         if res > best["map"]:
    #             best["map"] = res
    #             best["alpha"] = a
    #             best["beta"] = b
    #         print("Best", best)
    rec = CBFRecommender(dr)
    rec.fit(top_k=50,a=0.6,b=0.2)
    rec.recommend()
    ev = Evaluator()

    res = ev.evaluation(rec.eurm, rec.urm, dr, save=True, name="test")
