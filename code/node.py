import math
import re
from copy import copy

#log(x) = log(1 + x)
supported_complex_functions = [('exp', 1), ('sqrt', 1), ('log', 1), ('+', 2), ('*', 2), ('/', 2), ('1', 0)]


class NodeException(Exception):
    def __init__(self, message):
        self.message = message


class Node:
    def __init__(self, name, children):
        self.name = name
        self.children = copy(children)

    def count(self, numbers):
        children_results = [child.count(numbers) for child in self.children]
        if self.name == 'exp':
            return math.exp(children_results[0])
        elif self.name == 'sqrt':
            if children_results[0] < 0:
                return 0
            return children_results[0] ** 0.5
        elif self.name == 'log':
            if 1 + children_results[0] <= 0:
                return 0
            return math.log(1 + children_results[0])
        elif self.name == '+':
            return children_results[0] + children_results[1]
        elif self.name == '*':
            return children_results[0] * children_results[1]
        elif self.name == '/':
            if abs(children_results[1]) < 1e-02:
                return 0
            return children_results[0] / children_results[1]
        elif self.name == '1':
            return 1
        elif re.search('^#[0-9]+$', self.name):
            #return numbers.data[int(self.name[1:])]
            return numbers[0][int(self.name[1:])]
        else:
            raise NodeException("Unknown function")

    def __str__(self):
        children_results = ','.join([str(child) for child in self.children])
        result = ''.join([self.name, '(', children_results, ')'])
        return result


#be accurate! the same node can be in different trees


def generate_functions(max_depth, functions):
    generated_nodes = []

    for function, valance in functions:
        #print(function, valance)
        if max_depth == 1 and valance > 0:
            continue

        if valance == 0:
            generated_nodes.append(Node(function, []))
        else:
            children = []
            for i in range(valance):
                children_side = generate_functions(max_depth - 1, functions)
                children.append(children_side)

            cur = [0 for i in range(valance)]
            while True:
                generated_nodes.append(Node(function, [children[i][cur[i]] for i in range(valance)]))
                for i in range(valance - 1, -1, -1):
                    cur[i] += 1
                    if cur[i] == len(children[i]):
                        cur[i] = 0
                    else:
                        break

                end = True
                for i in range(valance):
                    if cur[i] != 0:
                        end = False
                if end:
                    break

    return generated_nodes


def main():
    functions = copy(supported_complex_functions)
    n = 2
    for i in range(n):
        functions.append((['#' + str(i), 0]))

    generated_functions = generate_functions(3, functions)
    for root in generated_functions:
        print(root)

if __name__ == '__main__':
    main()