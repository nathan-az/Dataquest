import numpy as np
import pandas as pd


def entropy(column):
    elements, counts = np.unique(column, return_counts=True)
    # if statement in comprehension stops nan result since 0*log2(x) is undefined, returns 0. in this case,
    # 1*log2(1) + 0*log2(0) = 0. zero entropy result, zero uncertainty is consistent with theory
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


def predict():
    return


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
        def run_id3(data, target, features, tree):
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
        def get_prediction(tree: object, row: object) -> object:
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
