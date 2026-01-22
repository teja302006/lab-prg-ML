import math
from collections import Counter

data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]

attributes = ['Outlook', 'Temperature', 'Humidity', 'Wind']

def entropy(data):
    labels = [row[-1] for row in data]
    counts = Counter(labels)
    ent = 0
    for count in counts.values():
        p = count / len(data)
        ent -= p * math.log2(p)
    return ent

def info_gain(data, attr_index):
    total_entropy = entropy(data)
    values = set(row[attr_index] for row in data)
    weighted_entropy = 0
    for value in values:
        subset = [row for row in data if row[attr_index] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)
    return total_entropy - weighted_entropy

def id3(data, attributes, attr_indices):
    labels = [row[-1] for row in data]
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    if not attr_indices:
        return Counter(labels).most_common(1)[0][0]
    gains = [(info_gain(data, i), i) for i in attr_indices]
    _, best_attr = max(gains)
    tree = {attributes[best_attr]: {}}
    values = set(row[best_attr] for row in data)
    for value in values:
        subset = [row for row in data if row[best_attr] == value]
        new_attr_indices = attr_indices[:]
        new_attr_indices.remove(best_attr)
        tree[attributes[best_attr]][value] = id3(subset, attributes, new_attr_indices)
    return tree

tree = id3(data, attributes, list(range(len(attributes))))

print("Decision Tree:")
print(tree)

def classify(tree, sample):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = sample[attr]
    return classify(tree[attr][value], sample)

new_sample = {
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'High',
    'Wind': 'Strong'
}

result = classify(tree, new_sample)
print("New Sample Classification:", result)
ID
