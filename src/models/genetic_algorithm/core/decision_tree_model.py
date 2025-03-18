from sklearn.tree import DecisionTreeClassifier

def get_decision_tree(random_state=42, max_depth=10):
    return DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)