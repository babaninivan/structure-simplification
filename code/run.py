from evaluation import Evaluate
#import matplotlib.pyplot as plt
import load
import numpy as np
import sys


def run_evaluation(dataset_path, predictors, additional_roots=None, max_number_of_queries=None, folds_num=5,
                   evaluation_functions=(('precision', 1), ('precision', 3), ('precision', 5), ('ndcg', 1),
                                         ('ndcg', 3), ('ndcg', 5), ('dcg', 1), ('dcg', 3), ('dcg', 5))):

    evaluation_results = [np.zeros(len(evaluation_functions)) for i in range(len(predictors))]

    for fold in load.load_dataset(dataset_path, additional_roots, max_number_of_queries, folds_num):
        (x_train, y_train, id_train), (x_test, y_test, id_test) = fold

        for index_predictor, predictor in enumerate(predictors):
            # sys.stderr.write(predictor.get_name() + '\n')
            # sys.stderr.flush()

            y_pred = predictor.learn_predict(x_train, y_train, x_test)

            for index_function, (func_type, rank) in enumerate(evaluation_functions):
                evaluation_results[index_predictor][index_function] += Evaluate.mean(func_type, rank,
                                                                                     y_test, y_pred, id_test)

    evaluation_results = [result / folds_num for result in evaluation_results]
    return evaluation_results