import trees as t

file = open('lenses.txt')
dataSet = [i.strip().split("\t") for i in file.readlines()]
labels = ["age","prescript","astigmatic","tearRate"]
tree = t.createTree(dataSet,labels.copy())
print(tree)