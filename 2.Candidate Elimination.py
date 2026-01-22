import copy
data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High',   'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High',   'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High',   'Strong', 'Cool', 'Change', 'Yes']
]

num_attr = len(data[0]) - 1
S = ['0'] * num_attr
G = [['?'] * num_attr]

for instance in data:
    x = instance[:-1]
    label = instance[-1]
    if label == 'Yes':
        for i in range(num_attr):
            if S[i] == '0':
                S[i] = x[i]
            elif S[i] != x[i]:
                S[i] = '?'
        G = [g for g in G if all(g[i] == '?' or g[i] == S[i] for i in range(num_attr))]
    else:
        new_G = []
        for g in G:
            for i in range(num_attr):
                if g[i] == '?' and S[i] != x[i]:
                    h = copy.deepcopy(g)
                    h[i] = S[i]
                    new_G.append(h)
        G = new_G
    print("S:", S)
    print("G:", G)
    print("------------------")

print("\nFinal Output")
print("Specific Boundary (S):", S)
print("General Boundary (G):", G)
