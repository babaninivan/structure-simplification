import numpy as np
from math import log


class Evaluate(object):
    @staticmethod
    def dcg(relevances, rank):
        if len(relevances) < rank:
            raise IndexError()

        discounts = np.log(np.arange(rank) + 1) / log(2.0)
        discounts[0] = 1
        cut_relevances = np.asarray(relevances[:rank])

        return np.sum((2 ** cut_relevances - 1) / discounts)

    @staticmethod
    def ndcg(relevances, rank):
        z = Evaluate.dcg(sorted(relevances, reverse=True), rank)
        if z == 0:
            return 0
        return Evaluate.dcg(relevances, rank) / z

    @staticmethod
    def precision(relevances, rank):
        if len(relevances) < rank:
            raise IndexError()

        return np.sum(np.fmin(relevances[:rank], np.ones(rank))) / rank

    @staticmethod
    def mean(func, rank, y_true, y_pred, query_ids):
        if func == 'precision':
            func = Evaluate.precision
        elif func == 'ndcg':
            func = Evaluate.ndcg
        elif func == 'dcg':
            func = Evaluate.dcg

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        query_ids = np.asarray(query_ids)

        #assume that query_ids are sorted

        scores = []
        previous_qid = query_ids[0]
        previous_loc = 0
        for loc, qid in enumerate(query_ids):
            if previous_qid != qid:
                if loc == 40:
                    x = 1
                chunk = slice(previous_loc, loc)
                ranked_relevances = y_true[chunk][np.argsort(y_pred[chunk])[::-1]]
                scores.append(func(ranked_relevances, rank))
                previous_loc = loc
            previous_qid = qid

        chunk = slice(previous_loc, len(query_ids))
        ranked_relevances = y_true[chunk][np.argsort(y_pred[chunk])[::-1]]
        scores.append(func(ranked_relevances, rank))

        return np.mean(scores)

    @staticmethod
    def str_mean(func, rank):
        res = ''
        if func == 'precision':
            res = 'Precision'
        elif func == 'ndcg':
            res = 'NDCG'
        elif func == 'dcg':
            res = 'DCG'
        return ''.join([res, '@', str(rank)])


if __name__ == '__main__':
    print(Evaluate.precision(relevances=[4, 1, 0, 1, 0, 0, 0], rank=5))