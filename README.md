# excercise 2 
#Why does moons dataset fail?

The Perceptron can only learn linearly separable data.
Moons dataset contains non-linear boundaries ,the two "moons" are curved and overlapping. The Perceptron draws a straight line (linear decision boundary), so it can't correctly classify much of the data.

# excercise 3
 Type          Cause                            Symptoms                                                     
 Underfitting  Model too simple (C too small)   Low training & test accuracy, boundary too smooth            
 Overfitting   Model too complex (C too large)  High training accuracy, lower test accuracy, wiggly boundary 
              
# excercise 4
Low C = softer margin : more margin violations allowed,simpler model ,better generalization
High C = stricter margin :fewer margin violations allowed ,fits training data more closely ,less generalization, risk of overfitting

# excercise 9
Linear Data: Iris Petal Length/Width
Best classifiers: Logistic Regression, Linear SVM, Perceptron
These perform well because the classes are mostly linearly separable in 2D

Nonlinear Data :Moons Dataset
Best classifiers: SVM with RBF kernel, KNN, Random Forest
These handle complex, curved boundaries better than linear classifiers

| Classifier    | Accuracy (Iris) | Accuracy (Moons) |
| ------------- | --------------- | ---------------- |
| Perceptron    | 0.73            | ~0.50           |
| Logistic Reg  | 0.53            | ~0.60           |
| SVM (linear)  | 0.26            | ~0.55           |
| Decision Tree | 0.895           | 0.85             |
| Random Forest | 0.95            | 0.90+            |
| KNN           | 0.915           | 0.88             |
