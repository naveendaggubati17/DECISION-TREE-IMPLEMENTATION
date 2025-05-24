# DECISION-TREE-IMPLEMENTATION

COMPANY : CODTECH IT SOLUTIONS

NAME : NAVEEN DAGGUBATI

INTERN ID : CT06DL790

DOMAIN : MACHINE LEARNING

DURATION : 6 WEEKS

MENTOR : NEELA SANTOSH


---


EXPLANATION OF THE CODE:


The code is structured to perform several key tasks: loading the dataset, preprocessing the data, splitting it into training and testing sets, training a decision tree classifier, making predictions, evaluating the model's performance, and finally visualizing the decision tree. Each of these steps is crucial for building a machine learning model that can effectively classify outcomes based on input features.

---

Step 1: Import Necessary Libraries

The first step in the code is to import the required libraries. The libraries used include:

Pandas: This library is essential for data manipulation and analysis. It provides data structures like DataFrames, which are ideal for handling tabular data.
NumPy: A library for numerical operations in Python. It is often used for handling arrays and performing mathematical operations.
Matplotlib: A plotting library used for data visualization. It allows us to create static, animated, and interactive visualizations in Python.
Scikit-Learn: A powerful machine learning library that provides tools for model training, evaluation, and various algorithms, including decision trees.

---

Step 2: Load the Dataset

The dataset is loaded from a specified URL using the `pd.read_csv()` function. This function reads a comma-separated values (CSV) file into a DataFrame. The dataset in question contains information about various drugs and their effects based on different features. By loading the dataset into a DataFrame, we can easily manipulate and analyze the data.

---

Step 3: Explore the Dataset

Before proceeding with the model building, it is often useful to explore the dataset. I explored the dataset from Kaggle and downloaded it. The code includes optional print statements that display the first few rows of the DataFrame using `df.head()`, the structure of the DataFrame with `df.info()`, and descriptive statistics with `df.describe()`. This exploration helps in understanding the data types, the presence of missing values, and the distribution of numerical features.

---

Step 4: Prepare the Data

In this step, the code prepares the data for modeling. The target variable, which is the outcome we want to predict (in this case, the type of drug), is separated from the features. The features are stored in `X`, while the target variable is stored in `y`. 

Since the dataset may contain categorical variables (e.g., gender, drug type), these need to be converted into a numerical format that the decision tree algorithm can understand. The `pd.get_dummies()` function is used to convert categorical variables into dummy/indicator variables. This process creates binary columns for each category, allowing the model to interpret these variables correctly.

---

Step 5: Split the Dataset

The dataset is then split into training and testing sets using the `train_test_split()` function from Scikit-Learn. This function randomly divides the data into two subsets: one for training the model and the other for testing its performance. In this case, 80% of the data is used for training, and 20% is reserved for testing. The `random_state` parameter ensures that the split is reproducible.

---

Step 6: Initialize and Train the Decision Tree Classifier

A `DecisionTreeClassifier` is initialized with a specified random state for reproducibility. The classifier is then trained using the `fit()` method, which takes the training features (`X_train`) and the corresponding target labels (`y_train`). During this training phase, the decision tree algorithm learns the patterns and relationships between the features and the target variable.

---

Step 7: Make Predictions

Once the model is trained, it can be used to make predictions on the test set. The `predict()` method is called with the test features (`X_test`), and the predicted outcomes are stored in `y_pred`. This step is crucial for evaluating how well the model generalizes to unseen data.

---

Step 8: Evaluate the Model

The accuracy of the model is evaluated using the `accuracy_score()` function from Scikit-Learn. This function compares the predicted outcomes (`y_pred`) with the actual outcomes (`y_test`) and calculates the proportion of correct predictions. The accuracy score is printed to the console, providing a quick assessment of the model's performance.

---

Step 9: Visualize the Decision Tree

Finally, the decision tree is visualized using the `plot_tree()` function from Scikit-Learn. This function generates a graphical representation of the decision tree, showing how decisions are made based on the features. The `filled=True` parameter colors the nodes based on the predicted class, while `feature_names` and `class_names` provide labels for the features and target classes, respectively. The visualization helps in understanding the decision-making process of the model and can be useful for interpreting the results.

---

Conclusion

In summary, this code provides a comprehensive workflow for building and visualizing a decision tree model using Scikit-Learn

---

OUTPUT:

![Image](https://github.com/user-attachments/assets/459a63df-7f89-423e-b69b-d3a2139c9dbf)
