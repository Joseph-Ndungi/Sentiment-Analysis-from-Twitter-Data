import pandas as pd

# Clean Data
data = pd.read_csv("Variables3.csv")

def tag_location(location):
    if location:
        return "Tagged"
    else:
        return "Non tagged"

# Drop variables
clean_data = data
#clean_data["Text"] = pd.to_numeric(clean_data["Text"])
clean_data["VerifiedRetweet"] = pd.to_numeric(clean_data["VerifiedRetweet"])
# clean_data["Characters"] = pd.to_numeric(clean_data["Characters"])
clean_data["Verified"] = pd.to_numeric(clean_data["Verified"])
clean_data["Protected"] = pd.to_numeric(clean_data["Protected"])
#clean_data["Location"] = pd.to_numeric(clean_data["Location"])
# clean_data["Location"] = ["TAGGED" if i >= 2 else "NON-TAGGED" for i in clean_data["Location"]]  # set geotags to true or false
# clean_data["Followers"] = [1 if i >= 500 else 0 for i in clean_data["Followers"]]  # if number of followers is greater than 500 then TRUE
clean_data["VerifiedRetweet"].fillna(1, inplace=True)  # Remove NA values
# clean_data.fillna(1, inplace=True)  # Convert NA values to 1

# print(clean_data.head())
# clean_data["Location"] = clean_data["Location"].apply(tag_location)

def tag_location(location):
    if isinstance(location, str) and location.strip():
        return 'Tagged'
    else:
        return 'Non Tagged'
    

clean_data["Location"] = clean_data["Location"].apply(tag_location)

#check the followers column for values greater than 500
def tag_followers(followers):
    if followers >= 500:
        return 1
    else:
        return 0
    
clean_data["Followers"] = clean_data["Followers"].apply(tag_followers)

clean_data.fillna(1, inplace=True)  # Convert NA values to 1

# SUBSETTING DATA TO TRAIN& TEST SETS
# Split data into train and test sets
#train = clean_data.sample(frac=0.8, random_state=1)

import pandas as pd
import numpy as np

def create_train_test(data, size=0.8, train=True):
    n_row = data.shape[0]
    total_row = int(size * n_row)
    train_sample = np.arange(0, total_row)
    if train:
        return data.iloc[train_sample,:]
    else:
        return data.iloc[-train_sample,:]

clean_data1 = clean_data.drop(["Protected"], axis=1) # remove unwanted variables
data_train = create_train_test(clean_data1, 0.8, train=True)
data_test = create_train_test(clean_data1, 0.8, train=False)
print(data_train.shape) # check dimensions for train data
print(data_test.shape) # check dimensions for test data

# print(data_train.head())

# Model Training
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(data_train.drop(columns=['Location']), data_train['Location'])

#Predict the response for test dataset
y_pred = clf.predict(data_test.drop(columns=['Location']))

# Model Evaluation
print("Accuracy:", accuracy_score(data_test['Location'], y_pred))
print("Confusion Matrix:", confusion_matrix(data_test['Location'], y_pred))
print("Classification Report:", classification_report(data_test['Location'], y_pred))

# Visualize Decision Tree
from sklearn import tree
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf,
                     feature_names=data_train.drop(columns=['Location']).columns,
                        class_names=data_train['Location'].unique(),
                        filled=True)

# Save Decision Tree
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)






# import pandas as pd
# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# import matplotlib.pyplot as plt
# from sklearn.tree import export_text

# #data_train = pd.read_csv('data_train.csv') # read in your data file

# X_train = data_train.drop(columns=['Location']) # features
# y_train = data_train['Location'] # target variable


