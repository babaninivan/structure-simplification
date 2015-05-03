import numpy as np
from sklearn.datasets import load_svmlight_file
import os.path


def load_file(file_path):
    return load_svmlight_file(file_path, dtype=np.float64, query_id=True)


def load_fold(fold_path):
    """Returns tuples x, y, query_id for train set and for test set"""

    return [load_file(os.path.join(fold_path, 'train.txt')), load_file(os.path.join(fold_path, 'test.txt'))]


def load_dataset(dataset_path, additional_roots, max_number_of_queries=None, fold_num=5):
    """Returns a list of pairs of sets: train set is first and test set is second"""

    result = []
    for i in range(1, fold_num + 1):
        result.append(load_fold(os.path.join(dataset_path, ''.join(['Fold', str(i)]))))

    for i in range(fold_num):
        for j in range(2):
            temp = list(result[i][j])
            temp[0] = temp[0].todense()
            if max_number_of_queries is not None:
                if max_number_of_queries < temp[0].shape[0]:
                    for t in range(3):
                        temp[t] = temp[t][range(max_number_of_queries)]

            if additional_roots is not None:

                base = []
                for t in range(temp[0].shape[0]):
                    base.append(temp[0][t].getA().tolist())
                for t in range(temp[0].shape[0]):
                    for root in additional_roots:
                        base[t][0].append(root.count(base[t]))
                    base[t] = base[t][0]
                    #base[t] = np.matrix(np.asarray(base[t]))

                temp[0] = np.matrix(base)

            result[i][j] = tuple(temp)
        result[i] = tuple(result[i])

    return result


if __name__ == '__main__':
    import node
    functions = [('#42', 0)]
    roots = node.generate_functions(1, functions)
    print(len(roots))

    tmp = load_dataset('./MQ2007', additional_roots=roots, fold_num=1, max_number_of_queries=1)
    print(tmp[0][0][0])
