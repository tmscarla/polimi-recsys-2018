"""
l'evaluator ora ha queste tipologie di chiamata:

- ev.evaluate_dict( dict_tua_soluzione )


example:

    ev = Evaluator()

    prova_da_valutare = pd.read_csv( "una_soluzione_a_caso.csv" )
    dict_tua_sol = ev.csv_to_dict(prova_da_valutare)

    # puoi o fare un dizionario delle soluzioni mentre costruisci la tua soluzion
    # o fare come le due istruzioni precedenti se da CSV

    ev.evaluate_dict( dict_tua_sol )

"""

import numpy as np
import scipy.sparse as sps
import pandas as pd
import re


class Evaluator(object):

    def __init__(self):
        test = pd.read_csv("data/solution.csv")

        self.dict_soluzione = self.csv_to_dict(test)

    def extract_songs(self, string_songs):
        if isinstance(string_songs, str):
            tmp = re.sub(' +', ' ', string_songs.rstrip())
            ids = tmp.split(' ')
            for i, el in enumerate(ids):
                ids[i] = int(el)
            return ids

        elif isinstance(string_songs, type(np.nan)):
            return []
        else:
            print("la soluzione ha dentro qualcosa di strano. exiting")
            exit()

    def csv_to_dict(self, df):
        pl_ids =  df['playlist_id'].values
        track_ids = [self.extract_songs(x) for x in df['track_ids'].values]
        return dict(zip(pl_ids,track_ids))


    def apk(self, actual, predicted, k=10):
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), k)


    def mapk(self, actual, predicted, k=10):
        return np.mean([self.apk(a, p, k) for a, p in zip(actual, predicted)])


    def evaluate_dict(self, dict_to_evaluate):

        if set(self.dict_soluzione.keys()) == set(dict_to_evaluate.keys()):

            predicted = list()
            actual = list()
            for key, value in self.dict_soluzione.items():
                actual.append( value )
                predicted.append(dict_to_evaluate[key] )

            res = self.mapk(actual,predicted)

            return res

        else:
            print("your solution dict hasn't got the right solution keys")
            exit()

    def csv(self):
        df = pd.DataFrame(self.prediction, columns=['playlist_id', 'track_ids'])
        df.to_csv(str(self) + '.csv', sep=',', index=False)
    def evaluation(self, eurm, urm, dr, save=False, name="no_name"):
        # Seed removing
        eurm = eurm - urm
        # Taking top 10
        prediction = []

        for row in dr.target_playlists:
            val = eurm.data[eurm.indptr[row]:eurm.indptr[row + 1]]
            ind = val.argsort()[-10:][::-1]
            ind = list(eurm[row].indices[ind])

            recommended_tracks_str = ' '.join([str(i) for i in ind])
            prediction.append([row, recommended_tracks_str])

        rec_df = pd.DataFrame(prediction, columns=['playlist_id', 'track_ids'])
        dict_tua_sol = self.csv_to_dict(rec_df)
        if save:
            rec_df.to_csv(str(self) + '.csv', sep=',', index=False)
        return self.evaluate_dict(dict_tua_sol)

if __name__ == '__main__':

    prova_da_valutare = pd.read_csv("CFRec.csv")

    ev = Evaluator()

    dict_prova = ev.csv_to_dict(prova_da_valutare)

    print(ev.evaluate_dict(dict_prova) )
