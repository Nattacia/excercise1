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
