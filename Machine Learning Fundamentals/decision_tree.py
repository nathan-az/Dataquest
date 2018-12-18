# Below is my own non-binary decision tree implementation. Non-binary was chosen to better suit predictions made on
# categorical data. I believe other solutions such as an adapted one-for-all binary split may function similarly.

# While I recognise a solution utilising hash tables will have lower runtimes in situations when a feature
# contains many unique values, this was simply a self-imposed challenge in OOP and class-based recursive solutions.

import numpy as np
import pandas as pd


def entropy(column):
    elements, counts = np.unique(column, return_counts=True)
    # "if" statement in list comprehension stops nan result. Since 0*log2(x) is undefined, we return 0. In this case,
    # 1*log2(1) + 0*log2(0) = 0. Zero entropy result == zero uncertainty, consistent with theory
    entropy = np.sum(
        [-(counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) if counts[i] > 0 else 0 for i in
         range(len(counts))])
    return entropy


def information_gain(data, split_name, target_name):
    target_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_name], return_counts=True)
    weighted_entropy = np.sum(
        [counts[i] / np.sum(counts) * entropy(data.loc[data[split_name] == vals[i], target_name]) for i in
         range(len(counts))])
    return target_entropy - weighted_entropy


def find_best(data, target_name, features):
    max_gain = 0
    best_col = ""
    for col in features:
        gain = information_gain(data, col, target_name)
        if gain > max_gain:
            max_gain = gain
            best_col = col
    return best_col


class Tree():
    def __init__(self, children: object = None, label: object = None, value: object = None) -> object:
        self.children = children if children is not None else []
        self.label = label
        self.value = value


class DecisionTree():
    tree: Tree

    def __init__(self):
        pass

    def fit(self, data, target, features):
        '''
        :param data:
        :param target:
        :param features:
        :return:

        The fit method is called on the DecisionTree.tree attribute. This field will ALWAYS be reset to an empty
        tree when fit is called to avoid issues when calling fit twice with different data.
        '''
        def run_id3(data: object, target: object, features: object, tree: object) -> object:
            '''
            (DataFrame, String, List, Tree) -> DataFrame[target].dtypes
            :param data:
            :param target:
            :param features:
            :return:

            The run_id3 function recursively splits on the best feature, saving that feature as the node label and
            creating child nodes with their values as the feature's unique values. The child node is passed as the
            "tree" parameter, and the filtered DataFrame as the data parameter. Once the repeatedly split DataFrame
            contains only one unique value, the node label is set to the Target name, and only one child is created
            with the predicted value.
            '''
            unique_targets = pd.unique(data[target])
            if len(unique_targets) == 1:
                tree.label = target
                tree.children.append(Tree(value=unique_targets[0]))
                return
            best_split = find_best(data, target, features)
            tree.label = best_split
            for unique_val in pd.unique(data[best_split]):
                new_tree = Tree()
                new_tree.value = unique_val
                tree.children.append(new_tree)
                run_id3(data[data[best_split] == unique_val], target, features, new_tree)

        self.tree = Tree()
        run_id3(data, target, features, self.tree)

    def predict(self, row):
        """
        :param row:
        :return:

        As with the fit method, the predict method is called on the DecisionTree.tree attribute after the fit method
        has populated the DecisionTree.
        """
        def get_prediction(tree: object, row: object) -> object:
            """
            :param tree:
            :param row:
            :return:

             The predict function recursively steps down the decision tree, checking each child's value attribute for
            that which matches the data. Once a node is found with only one child (indicating no further splits), the
            child's value (prediction) is returned.
            """
            column = tree.label
            if len(tree.children) == 1:
                return tree.children[0].value
            for i in range(len(tree.children)):
                # if tree.children[i] is not None:
                if tree.children[i].value == row[column]:
                    return get_prediction(tree.children[i], row)

        return get_prediction(self.tree, row)


def testing_pred():
    # testing data duplicated from https://www.youtube.com/watch?v=eKD5gxPPeY0 -- thank you!

    outlook = ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain', 'Sunny',
               'Overcast', 'Overcast', 'Rain', 'Rain']
    humidity = ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'Normal', 'Normal',
                'High',
                'Normal', 'High', 'High']
    wind = ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
            'Weak', 'Strong', 'Weak']
    play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No', '?']

    columns = ["Outlook", "Humidity", "Wind", "Play"]
    data = pd.DataFrame([outlook, humidity, wind, play]).T
    data.columns = columns
    train = data.iloc[:-1, :]
    test = data.iloc[-1, :3]
    features = columns.copy()
    features.remove("Play")
    target = "Play"

    dt = DecisionTree()
    dt.fit(train, target, features)
    pred = dt.predict(test)
    return pred


if __name__ == "__main__":
    print("Prediction for 'Play' on testing data: {}".format(testing_pred()))

