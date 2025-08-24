from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from graphviz import Source as graph_from_dot_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pydotplus import graph_from_dot_data
import matplotlib
matplotlib.use('TkAgg')  # Or 'QtAgg' if you're on Windows  


# Removed undefined 'Source' usage; tree visualization is handled below with graph_from_dot_data

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx], label=cl)

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='none', alpha=1.0,
                    linewidth=1, marker='o', s=100, label='test set')
    plt.legend(loc='upper left')

iris = load_iris()
X = iris.data[:, [2, 3]]  # Petal length and width
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

sc_m = StandardScaler()
X_train_std = sc_m.fit_transform(X_train)
X_test_std = sc_m.transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

for depth in range(1, 11):
    tree = DecisionTreeClassifier(criterion='gini', max_depth=depth, random_state=1)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Depth={depth}, Accuracy={acc:.3f}")

    # Plot decision region
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))

    plt.figure()
    plot_decision_regions(X_combined, y_combined, classifier=tree,
                          test_idx=range(len(X_train), len(X_combined)))
    plt.title(f"Decision Tree (depth={depth})")
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.tight_layout()
    plt.show()

    # Export tree diagram
    dot_data = export_graphviz(
    tree,
    filled=True,
    rounded=True,
    class_names=['Setosa', 'Versicolor', 'Virginica'],
    feature_names=['petal length', 'petal width'],
    out_file=None
)

graph = graph_from_dot_data(dot_data)
graph.write_png(f'tree_depth_{depth}.png')