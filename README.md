# Decision-Tree-Classifier

### Objective:
The goal of this project is to design, implement, and evaluate a Decision Tree Classifier from scratch without relying on external machine learning libraries. Decision trees are fundamental components of many machine learning algorithms, and creating one from scratch will deepen our understanding of the underlying principles and algorithms governing decision tree construction.

### Scope:
This project will involve creating a Decision Tree Classifier algorithm to perform binary classification. The scope includes defining the structure of the tree, implementing the algorithms for building the tree, and incorporating mechanisms for making predictions based on input features.

### Pseudo-Code
Below is the Pseudo-code for decision tree classifier:

1. Fit the data received on X and Y, then proceed to instantiate the next function known as `generateTree`. Within this function, the data is divided into features and labels, and subsequently passed to the function named `bestSplitter`.

2. `bestSplitter` is responsible for splitting the left and right leaf based on a given criterion, utilizing thresholds determined by the `thresholdMidpoints()` method. Specifically, for the Gini criterion, the information gain is computed through the `giniScore` method. Likewise, for the entropy criterion, information gain is determined using the `entropyScore` method.

3. Upon completion of these calculations, the values are returned to the `generateTree` function, which performs a recursive calculation for both the left and right branches of the tree. The calculated values are then returned to the `fit` function and stored in the root variable.

4. Subsequently, predictions for test values are made based on the root of the decision tree.
