import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import joblib

# Define DecisionTree and RandomForest classes
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, data, target, features, depth=0):
        if depth == self.max_depth or len(np.unique(data[target])) == 1:
            most_common_label = Counter(data[target]).most_common(1)[0][0]
            return most_common_label

        best_feature = features[0]
        tree = {best_feature: {}}

        for value in np.unique(data[best_feature]):
            subset = data[data[best_feature] == value]
            if subset.empty:
                most_common_label = Counter(data[target]).most_common(1)[0][0]
                tree[best_feature][value] = most_common_label
            else:
                tree[best_feature][value] = self.fit(subset, target, [f for f in features if f != best_feature], depth + 1)

        self.tree = tree
        return tree

    def predict_row(self, row, tree):
        if not isinstance(tree, dict):
            return tree

        feature = next(iter(tree))
        if row[feature] in tree[feature]:
            return self.predict_row(row, tree[feature][row[feature]])
        else:
            return Counter(data[target]).most_common(1)[0][0]

    def predict(self, data):
        return data.apply(lambda row: self.predict_row(row, self.tree), axis=1)

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, sample_size=0.8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.sample_size = sample_size
        self.trees = []

    def bootstrap_sample(self, data):
        indices = np.random.choice(data.index, size=int(len(data) * self.sample_size), replace=True)
        return data.loc[indices]

    def fit(self, data, target, features):
        for _ in range(self.n_trees):
            sample = self.bootstrap_sample(data)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(sample, target, features)
            self.trees.append(tree)

    def predict(self, data):
        tree_preds = np.array([tree.predict(data) for tree in self.trees])
        majority_votes = [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(data.shape[0])]
        return majority_votes

# Streamlit App
def main():
    st.title("E-commerce Customer Behavior Prediction")
    st.write("Upload a CSV file and predict customer satisfaction.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file)
        st.write("Preview of the dataset:")
        st.dataframe(data.head())

        if st.button("Train and Predict"):
            target = 'Satisfaction Level'
            selected_features = ['Gender', 'Membership Type', 'Total Spend', 'Discount Applied', 'City']

            # Check for required columns
            missing_cols = [col for col in selected_features + [target] if col not in data.columns]
            if missing_cols:
                st.error(f"Missing columns in dataset: {missing_cols}")
                return

            # Split data
            def split_data(data, test_size=0.2, random_state=None):
                if random_state:
                    np.random.seed(random_state)
                indices = np.arange(len(data))
                np.random.shuffle(indices)
                split_point = int(len(data) * (1 - test_size))
                train_indices, test_indices = indices[:split_point], indices[split_point:]
                return data.iloc[train_indices], data.iloc[test_indices]

            train_data, test_data = split_data(data, test_size=0.2, random_state=23)

            # Train Random Forest
            rf = RandomForest(n_trees=10, max_depth=5)
            rf.fit(train_data, target, selected_features)

            # Predict
            predictions = rf.predict(test_data)

            # Calculate accuracy
            accuracy = np.mean(predictions == test_data[target])
            st.success(f"Model Accuracy: {accuracy:.2f}")

            # Show predictions
            test_data = test_data.copy()
            test_data['Prediction'] = predictions
            st.write("Predictions:")
            st.dataframe(test_data)

if __name__ == "__main__":
    main()
