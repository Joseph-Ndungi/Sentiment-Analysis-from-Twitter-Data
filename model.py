import pandas as pd
import numpy as np

# Clean Data
clean_data = pd.read_csv("Variables3.csv")


# def tag_status(booleanStatus):
#     if booleanStatus == True:
#         return 1
#     else:
#         return 0
    
# clean_data["Verified"] = clean_data["Verified"].apply(tag_status)
# clean_data["Protected"] = clean_data["Protected"].apply(tag_status)
# clean_data["VerifiedRetweet"] = clean_data["VerifiedRetweet"].apply(tag_status)
clean_data["VerifiedRetweet"].fillna(1, inplace=True)  # Remove NA values


def tag_location(location):
    if isinstance(location, str) and location.strip():
        return "Tagged"
    else:
        return "Non-Tagged"
    

clean_data["Location"] = clean_data["Location"].apply(tag_location)

#check the followers column for values greater than 500
def tag_followers(followers):
    if followers >= 500:
        return True
    else:
        return False
    
clean_data["Followers"] = clean_data["Followers"].apply(tag_followers)

#count characters in the text column
def count_characters(text):
    if isinstance(text, str):
        return len(text)
    else:
        return 0
    
clean_data["Character"] = clean_data["Text"].apply(count_characters)

def tag_characters(characters):
    if characters >= 100:
        return True
    else:
        return False


clean_data["Character"] = clean_data["Character"].apply(tag_characters)



clean_data.fillna(1, inplace=True)  # Convert NA values to 1


def create_train_test(data, size=0.8, train=True):
    n_row = data.shape[0]
    total_row = int(size * n_row)
    train_sample = np.arange(0, total_row)
    if train:
        return data.iloc[train_sample,:]
    else:
        return data.iloc[-train_sample,:]

clean_data1 = clean_data.drop(["Text"], axis=1) # remove unwanted variables
clean_data1 = clean_data1.drop(["Source"], axis=1) # remove unwanted variables
clean_data1 = clean_data1.drop(["Unnamed: 0"], axis=1) # remove unwanted variables

#move the target variable to the last column
cols = list(clean_data1.columns.values)
cols.pop(cols.index('Location'))
clean_data1 = clean_data1[cols+['Location']]
print(clean_data1.head())
# data_train = create_train_test(clean_data1, 0.8, train=True)
# print(data_train.shape) # check dimensions for train data


# from sklearn.tree import DecisionTreeClassifier, plot_tree
# import matplotlib.pyplot as plt


# define features and target variable
# features = data_train.columns[:-1] # all columns except the last one
# target = data_train.columns[-1] # last column



# # create and fit the decision tree model
# model = DecisionTreeClassifier()
# model.fit(data_train[features], data_train[target])

# # visualize the decision tree using plot_tree
# plt.figure(figsize=(20,10))
# plot_tree(model, filled=True, feature_names=features, class_names=model.classes_)
# plt.show()

X = clean_data1[['Verified', 'Protected', 'Followers', 'VerifiedRetweet', 'Character']]
y = clean_data1['Location']


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # create a decision tree model
# dt = DecisionTreeClassifier(random_state=42)

# # define the hyperparameters and their possible values
# param_grid = {
#     'criterion': ['gini', 'entropy'],
#     'max_depth': [3, 5, 7, 10],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# # perform grid search to find the best hyperparameters
# grid_search = GridSearchCV(dt, param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# # print the best hyperparameters and the corresponding score
# print('Best hyperparameters:', grid_search.best_params_)
# print('Best score:', grid_search.best_score_)

# # fit the model using the best hyperparameters
# dt_best = grid_search.best_estimator_
# dt_best.fit(X_train, y_train)

# # evaluate the performance of the model on the test set
# accuracy = dt_best.score(X_test, y_test)
# print('Accuracy:', accuracy)

# # visualize the decision tree using plot_tree
# plt.figure(figsize=(20,10))
# plot_tree(dt_best, filled=True, feature_names=X_train.columns, class_names=dt_best.classes_)
# plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced', 'balanced_subsample', None]
}

# Create a Random Forest classifier object
rf_model = RandomForestClassifier(random_state=42)

# Perform a grid search to find the best hyperparameters
grid_search = GridSearchCV(rf_model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best hyperparameters
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# Use the best model to make predictions on the test set
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

# Evaluate the performance of the model on the test set
accuracy = best_rf_model.score(X_test, y_test)
print('Accuracy:', accuracy)

# Print the confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))

# Print the classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

# Print the feature importances
print(best_rf_model.feature_importances_)

# Plot the feature importances
plt.figure(figsize=(10,5))
plt.bar(X_train.columns, best_rf_model.feature_importances_)
plt.show()

#save the model
import pickle
filename = 'finalized_model.sav'
pickle.dump(best_rf_model, open(filename, 'wb'))










