
# UFC-Predictor

The purpose of this project is to apply my passion for Machine Learning to the exciting world of MMA! MMA, as well as any other combat sport, is hard to consistently predict the victor due to the overwhelming amount of variables that can influence the outcome of a fight. However, I believe that it is worth wrangling the available data and leveraging machine learning to make ***statistically*** informed predictions. I am still working on the flow between Jupyter notebooks but it is currently designed as follows:

1. EDA, model creation, and training
2. Model evaluation on unseen data using common classification error metrics (confusion matrix/classification report)
3. Running predictions for upcoming main card using the trained model as well as the associated W/L probabilities

## Model Selection

Further analysis of the data must be conducted to acquire the features that appear most pertinent to winning. The current model used is a Support Vector Classifier which I implemented with Scikit Learn's amazing library. A support vector classifier attempts to separate data geometrically via an (n-1)-dimensional hyperplane in an n-dimensional feature space while limiting the amount of misclassifications. The math behind this process is quite intensive, however, with the help of Scikit-Learn, this process is as simple as:

```
from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)
```

Pretty neat!

When I first began working on this project I was familiar with Logistic Regression, KNN, and SVM. I decided to exploit SVM when I first began this project, however I would like to transition to a Random Forest Classifier. 

