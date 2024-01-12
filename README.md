# Decision-Tree-Classifier

Below is the psudo-code for decision tree classifier:

1. Fit the data which on the X and Y received and instantiate the next function, generateTree.
2. generateTree will take the data and divide it into feature and labels and pass to bestSplitter.
3. bestSplitter, splits the left and right leaf based on criterion using thresholds calculated by thresholdMidpoints().
4. For Gini criterion, information gain will be calculated using giniScore method.
5. Similarly, for entropy we will calculate information gain using entropyScore method.
6. Once done, return the values to fit generateTree and calculate recursively for left and right tree.
7. Finally returning to fit value and save it to root variable.
8. And predict test values based on the root.
