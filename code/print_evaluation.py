from prettytable import PrettyTable
from evaluation import Evaluate
from run import run_evaluation
from base_algorithm import PredictorGBR
from base_algorithm import PredictorFunction
import numpy as np
import node
from copy import copy
import sys


def make_sorted_list(algorithms, evaluation_functions, get_key,
                     additional_roots=None, max_number_of_queries=None, folds_num=5):
    result_evaluation = run_evaluation('./MQ2007', algorithms, evaluation_functions=evaluation_functions,
                                       additional_roots=additional_roots,
                                       max_number_of_queries=max_number_of_queries, folds_num=folds_num)
    for index, algorithm in enumerate(algorithms):
        result_evaluation[index] = [result_evaluation[index], algorithm]

    result_evaluation = sorted(result_evaluation, key=get_key)
    return result_evaluation


def make_table(ordered_algorithms, evaluation_functions):
    result = PrettyTable()

    result.add_column('Algorithm', [])
    for func_name in [Evaluate.str_mean(name, rank) for name, rank in evaluation_functions]:
        result.add_column(func_name, [])
    for info in ordered_algorithms:
        result.add_row([info[1].get_name()] + [x for x in info[0]])
    return result


class RedirectStdoutTo:
    def __init__(self, out_new):
        self.out_new = out_new

    def __enter__(self):
        self.out_old = sys.stdout
        sys.stdout = self.out_new

    def __exit__(self, *args):
        sys.stdout = self.out_old


FEATURES_NUMBER = 46


def delete_algorithms_with_same_results(ordered_algorithms):
    new_ordered_algorithms = []
    for i in range(len(ordered_algorithms)):
        if i == 0:
            new_ordered_algorithms.append(ordered_algorithms[i])
            continue

        equal = True
        for x, y in zip(ordered_algorithms[i - 1][0], ordered_algorithms[i][0]):
            if x != y:
                equal = False

        if not equal:
            new_ordered_algorithms.append(ordered_algorithms[i])

    return new_ordered_algorithms


def get_some_good_roots():
    functions_for_depth_2 = copy(node.supported_complex_functions)
    n = FEATURES_NUMBER
    for i in range(n):
        functions_for_depth_2.append((['#' + str(i), 0]))

    roots_depth_1 = node.generate_functions(1, functions_for_depth_2)
    roots_depth_2 = node.generate_functions(2, functions_for_depth_2)

    ordered_algorithms = make_sorted_list([PredictorFunction(root) for root in roots_depth_1],
                                          (('precision', 5), ('ndcg', 5)),
                                          lambda x: x[0][1], max_number_of_queries=1000, folds_num=1)

    MAGIC = 4
    best_features_names = [(algorithm.root.name, len(algorithm.root.children)) for temp, algorithm in ordered_algorithms[-MAGIC:]]

    functions_for_depth_3 = copy(node.supported_complex_functions) + copy(best_features_names)
    roots_depth_3 = node.generate_functions(3, functions_for_depth_3)

    print(len(roots_depth_3))


    ordered_algorithms = make_sorted_list([PredictorFunction(root) for root in roots_depth_2 + roots_depth_3],
                                          (('precision', 5), ('ndcg', 5)),
                                          lambda x: x[0][1], max_number_of_queries=1000, folds_num=1)

    #
    # ordered_algorithms = make_sorted_list(best_feature, (('precision', 1), ('precision', 3), ('precision', 5),
    #                                                         ('ndcg', 1), ('ndcg', 3), ('ndcg', 5),
    #                                                         ('dcg', 1), ('dcg', 3), ('dcg', 5)),
    #                                       lambda x: x[0][1], max_number_of_queries=None, folds_num=5)

    ordered_algorithms = delete_algorithms_with_same_results(ordered_algorithms)
    #result = make_table(ordered_algorithms, (('precision', 5), ('ndcg', 5)))

    NEW_FEATURES_NUMBER = 46
    best_roots = [algorithm.root for temp, algorithm in ordered_algorithms[-NEW_FEATURES_NUMBER:]]

    return best_roots

def main():
    # result = make_table([PredictorGBR()], (('precision', 1), ('precision', 3), ('precision', 5),
    #                                        ('ndcg', 1), ('ndcg', 3), ('ndcg', 5),
    #                                        ('dcg', 1), ('dcg', 3), ('dcg', 5)))

    functions_for_depth_2 = copy(node.supported_complex_functions)
    n = FEATURES_NUMBER
    for i in range(n):
        functions_for_depth_2.append((['#' + str(i), 0]))

    best_roots = get_some_good_roots()

    #node.generate_functions(1, functions_for_depth_2)

    result1 = make_sorted_list([PredictorGBR(n_estimators=100), PredictorGBR(n_estimators=10)],
                               (('precision', 1), ('precision', 3), ('precision', 5),
                                ('ndcg', 1), ('ndcg', 3), ('ndcg', 5),
                                ('dcg', 1), ('dcg', 3), ('dcg', 5)), lambda x: x[0][1])

    result1 = make_table(result1, (('precision', 1), ('precision', 3), ('precision', 5),
                                ('ndcg', 1), ('ndcg', 3), ('ndcg', 5),
                                ('dcg', 1), ('dcg', 3), ('dcg', 5)))

    result2 = make_sorted_list([PredictorGBR(n_estimators=100), PredictorGBR(n_estimators=10)],
                               (('precision', 1), ('precision', 3), ('precision', 5),
                                ('ndcg', 1), ('ndcg', 3), ('ndcg', 5),
                                ('dcg', 1), ('dcg', 3), ('dcg', 5)), lambda x: x[0][1], additional_roots=best_roots)

    result2 = make_table(result2, (('precision', 1), ('precision', 3), ('precision', 5),
                                    ('ndcg', 1), ('ndcg', 3), ('ndcg', 5),
                                    ('dcg', 1), ('dcg', 3), ('dcg', 5)))

    with open('log.txt', 'w') as log, RedirectStdoutTo(log):
        print(result1)
        print(result2)


if __name__ == '__main__':
    main()