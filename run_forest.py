# Random Forest implementation using custom decision tree
import numpy as np
from pipeline_matrix import build_feature_matrix
from decision_tree import decision_tree_algorithm, decision_tree_predictions
from pipeline_matrix import collect_image_paths
import os

class RandomForestClassifierCustom:
    def __init__(self, n_estimators=10, min_samples=2, max_depth=5, random_subspace=None):
        self.n_estimators = n_estimators
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.random_subspace = random_subspace
        self.trees = []
        self.text_vectorizer = None

    def fit(self, X, y, feature_names):
        self.trees = []
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            # Bootstrap sample
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[idxs]
            y_sample = y[idxs]
            # Build DataFrame for decision_tree_algorithm
            import pandas as pd
            df = pd.DataFrame(X_sample, columns=feature_names)
            df['label'] = y_sample
            tree = decision_tree_algorithm(df, min_samples=self.min_samples, max_depth=self.max_depth, random_subspace=self.random_subspace)
            self.trees.append(tree)

    def predict(self, X, feature_names):
        import pandas as pd
        df = pd.DataFrame(X, columns=feature_names)
        # Get predictions from all trees
        tree_preds = np.array([
            decision_tree_predictions(df, tree) for tree in self.trees
        ])
        # Majority vote
        y_pred = []
        for i in range(X.shape[0]):
            votes = tree_preds[:, i]
            # Most common label
            vals, counts = np.unique(votes, return_counts=True)
            y_pred.append(vals[counts.argmax()])
        return np.array(y_pred)

if __name__ == "__main__":
    # Build feature matrix and labels
    X, y, text_vectorizer = build_feature_matrix('results')
    feature_names = [f'feat_{i}' for i in range(X.shape[1])]
    # Shuffle and split
    idxs = np.arange(len(y))
    np.random.shuffle(idxs)
    split = int(0.8 * len(y))
    train_idx, test_idx = idxs[:split], idxs[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    image_paths = collect_image_paths('results')
    test_image_paths = [image_paths[i] for i in test_idx]

    # Train Random Forest
    rf = RandomForestClassifierCustom(n_estimators=10, min_samples=2, max_depth=5, random_subspace=None)
    rf.fit(X_train, y_train, feature_names)

    # Predict
    y_pred = rf.predict(X_test, feature_names)
    acc = np.mean(y_pred == y_test)
    print(f"Random Forest accuracy: {acc:.3f}")

    # Define category mapping (edit as needed for your dataset)
    CATEGORY_NAMES = [
        "Produce",
        "Fruit",
        "Dairy",
        "Grains",
        "Meats"
    ]

    # If your labels are integers (0-4), map directly. If they are strings, map accordingly.
    # Example: if y contains integers 0-4
    def get_category(label):
        try:
            idx = int(label)
            return CATEGORY_NAMES[idx]
        except:
            # If label is string, map by name (customize as needed)
            label_map = {
                "produce": "Produce",
                "fruit": "Fruit",
                "dairy": "Dairy",
                "grains": "Grains",
                "meat": "Meats"
            }
            return label_map.get(label.lower(), label)

    print("\nImage classification results:")
    for img_path, pred_label in zip(test_image_paths, y_pred):
        print(f"{os.path.basename(img_path)} => {get_category(pred_label)}")
