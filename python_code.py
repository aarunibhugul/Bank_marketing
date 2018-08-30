#--------------------Importing the required libraries--------------#
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #or sklearn.preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

#----------------------Data Collection-----------------#

parent_dataset = pd.read_table('Business Case - SP Analytics - data.txt')

#---------------------Data Preprocessing---------------#

#--checking for null values--#

missing_columns_list = []
non_missing_columns_list = []

for col in parent_dataset.columns:
    if(parent_dataset[col].isnull().any()) == True:
        missing_columns_list.append(col)
    else:
        non_missing_columns_list.append(col)

#----processing (binning) age column-------#
bin = [17,27,37,47,57,67,77,87,97]
age_groups = ['18-27','28-37','38-47','48-57','58-67','68-77','78-87','88-97']

parent_dataset['age'] = pd.cut(parent_dataset['age'], bin, labels = age_groups)

#----processing pday column and converting it to months column----#

parent_dataset['pdays_not_contacted'] = 0
parent_dataset.loc[parent_dataset['pdays'] == -1 ,'pdays_not_contacted'] = 1
# converting the columns to month#
parent_dataset['months_passed'] = parent_dataset['pdays']/30
parent_dataset.loc[(parent_dataset['months_passed'] < 0), 'months_passed'] = 0
parent_dataset['months_passed'] = parent_dataset['months_passed'].astype(int)
parent_dataset['months_passed'] = parent_dataset['months_passed'].astype(str)
parent_dataset = parent_dataset.drop(['pdays'], axis = 1)


#processing 'days' column and converting it to weeks column#
parent_dataset['Week_contacted'] = "no data" # create a column dtype = str
parent_dataset.loc[(parent_dataset['day'] > 0) & (parent_dataset['day']<=7), 'Week_contacted'] = 'Week 1'
parent_dataset.loc[(parent_dataset['day'] >= 8) & (parent_dataset['day']<=14), 'Week_contacted'] = 'Week 2'
parent_dataset.loc[(parent_dataset['day'] >= 15) & (parent_dataset['day']<=21), 'Week_contacted'] = 'Week 3'
parent_dataset.loc[(parent_dataset['day'] >=22) & (parent_dataset['day']<=28), 'Week_contacted'] = 'Week 4'
parent_dataset.loc[(parent_dataset['day'] >= 29) & (parent_dataset['day']<=31), 'Week_contacted'] = 'Week 5'
parent_dataset = parent_dataset.drop(['day'], axis = 1)

#processing 'campaign' column#
bin_campaign = [0,3,6,9,12,60]
campaign_bins = ['1-3 times','4-6 times','7-9 times','10-12 times','more than 13 times']
parent_dataset['campaign contacts'] = pd.cut(parent_dataset['campaign'], bin_campaign, labels = campaign_bins)
parent_dataset = parent_dataset.drop(['campaign'], axis = 1)

#processing 'previous' column#
parent_dataset['Previous campaign contacts'] = "no data"
parent_dataset.loc[(parent_dataset['previous'] == 0), 'Previous campaign contacts'] = 'zero times'
parent_dataset.loc[(parent_dataset['previous'] == 1), 'Previous campaign contacts'] = 'once'
parent_dataset.loc[(parent_dataset['previous'] == 2), 'Previous campaign contacts'] = 'twice'
parent_dataset.loc[(parent_dataset['previous'] == 3), 'Previous campaign contacts'] = 'thrice'
parent_dataset.loc[(parent_dataset['previous'] == 4), 'Previous campaign contacts'] = '4 times'
parent_dataset.loc[(parent_dataset['previous'] >4), 'Previous campaign contacts'] = 'more than 4 times'
parent_dataset = parent_dataset.drop(['previous'], axis = 1)
# dropped 272

#processing the 'balance' columns#
parent_dataset["balance type"] = 'no data'
parent_dataset.loc[parent_dataset['balance']<0,'balance type'] = 'Negative Balance'
parent_dataset.loc[parent_dataset['balance']==0,'balance type'] = 'Zero Balance'
parent_dataset.loc[(parent_dataset['balance'] >= 1) & (parent_dataset['balance'] <= 500),'balance type'] = "1-500"
parent_dataset.loc[(parent_dataset['balance'] >= 501) & (parent_dataset['balance'] <= 1000),'balance type'] = "501 to 1000"
parent_dataset.loc[(parent_dataset['balance'] >= 1001) & (parent_dataset['balance'] <= 1500),'balance type'] = "1001 to 1500"
parent_dataset.loc[(parent_dataset['balance'] >= 1501) & (parent_dataset['balance'] <= 2000),'balance type'] = "1501 to 2000"
parent_dataset.loc[(parent_dataset['balance'] > 2000),'balance type'] = "more than 2000"
parent_dataset = parent_dataset.drop(['balance'], axis = 1)



#--------Model Building------#
#dividing into dataset into independent(x) and target(y) variables#

X = parent_dataset.drop(['y', 'duration'], axis = 1)
X = pd.get_dummies(X)
Numerical_encoding = {'yes':1, 'no':0}
y = parent_dataset["y"].apply(lambda x: Numerical_encoding[x])

#splitting into training and test#

from sklearn.model_selection import train_test_split #or sklearn.preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#Building DecisionTree Classifier to extract important features#
from sklearn.tree import DecisionTreeClassifier
decision_tree_classifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0)
decision_tree_classifier.fit(X_train, y_train)
y_pred = decision_tree_classifier.predict(X_test)

##Checking the accuracy using confusion_matrix##
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
sum_of_diagonal_elements = sum(np.diagonal(cm))
sum_of_all_elements_confusion_matrix = np.sum(cm)
Accuracy_score = sum_of_diagonal_elements/sum_of_all_elements_confusion_matrix
print("***DecisionTreeClassifier accuracies*****")
print("confusion_matrix score: ",Accuracy_score*100)

####Checking the accuracy Using Cross cross_validation####
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = decision_tree_classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
print("cross_val_score : ",accuracies.mean()*100)

##Checking the most Important feature##
decision_tree_classifier.fit(X, y)
feature_importances = pd.DataFrame([list(decision_tree_classifier.feature_importances_),list(X)], columns= list(X))
feature_importances = feature_importances.T
feature_importances.columns = ['importance','feature']
feature_importances = feature_importances.sort_values(by = ['importance'], ascending = False)

#-----Building Random Forest Classifier based on extracted features list-----#
extracted_features_list = feature_importances.iloc[:47,1]
X_random_forest = X[extracted_features_list]
Numerical_encoding = {'yes':1, 'no':0}
y_random_forest = parent_dataset["y"].apply(lambda x: Numerical_encoding[x])

from sklearn.model_selection import train_test_split #or sklearn.preprocessing
X_train, X_test, y_train, y_test = train_test_split(X_random_forest, y_random_forest, test_size = 0.25, random_state = 0)


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
sum_of_diagonal_elements = sum(np.diagonal(cm))
sum_of_all_elements_confusion_matrix = np.sum(cm)
Accuracy_score = sum_of_diagonal_elements/sum_of_all_elements_confusion_matrix
print("***RandomForestClassifier accuracies*****")
print("confusion_matrix score: ",Accuracy_score*100)



# Building LogisticRegression based on Extracted features#
X = X[extracted_features_list]
X = pd.get_dummies(X)

Numerical_encoding = {'yes':1, 'no':0}

y = parent_dataset["y"].apply(lambda x: Numerical_encoding[x])

from sklearn.model_selection import train_test_split #or sklearn.preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
sum_of_diagonal_elements = sum(np.diagonal(cm))
sum_of_all_elements_confusion_matrix = np.sum(cm)
Accuracy_score = sum_of_diagonal_elements/sum_of_all_elements_confusion_matrix
print("***LogisticRegression accuracies*****")
print("confusion_matrix score: ",Accuracy_score*100)


#--------Probability Calculation ----------#

X_yes_df = parent_dataset[parent_dataset['y']=='yes']
X_no_df = parent_dataset[parent_dataset['y']=='no']

y_yes_df = parent_dataset['y'][(parent_dataset['y'])=='yes']


Numerical_encoding = {'yes':1, 'no':0}
y_yes_df = y_yes_df.apply(lambda x: Numerical_encoding[x])
X_yes_df = X_yes_df.drop(['y', 'duration'], axis = 1)
X_yes_df = pd.get_dummies(X_yes_df)

print(len(X_yes_df['balance type_Zero Balance']))
print(len(y_yes_df))

X_no_df = X_no_df.drop(['y', 'duration'], axis = 1)
X_no_df = pd.get_dummies(X_no_df)


#----------Finding which column is least common among the X_yes_df and X dataframes----#
big_list  = X.columns

small_list = X_yes_df.columns

common_list = []
uncommon_list = []

for i in big_list:
    if i in small_list:
        common_list.append(i)
    else:
        uncommon_list.append(i)
print(common_list)
print(uncommon_list)

##-------Again logistic regression after emmitting 'months_passed_29' column-----#
X = parent_dataset.drop(['y', 'duration'], axis = 1)
X = pd.get_dummies(X)
X = X.drop(['months_passed_29'], axis = 1)
Numerical_encoding = {'yes':1, 'no':0}

y = parent_dataset["y"].apply(lambda x: Numerical_encoding[x])

from sklearn.model_selection import train_test_split #or sklearn.preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

proabability_prediction = classifier.predict_proba(X_yes_df)
print("yes")
print(proabability_prediction[:,0])
#-----------Designing Probability column for parent_dataset with y = 'Yes'-------#
parent_dataset_yes = parent_dataset[parent_dataset['y'] == 'yes']
parent_dataset_yes['Probability'] = list(proabability_prediction[:,0])

#-----------Designing Probability column for parent_dataset with y = 'no'-------#
X = parent_dataset.drop(['y', 'duration'], axis = 1)
X = pd.get_dummies(X)

Numerical_encoding = {'yes':1, 'no':0}

y = parent_dataset["y"].apply(lambda x: Numerical_encoding[x])

from sklearn.model_selection import train_test_split #or sklearn.preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

proabability_prediction = classifier.predict_proba(X_no_df)
parent_dataset_no = parent_dataset[parent_dataset['y'] == 'no']
proabability_prediction = classifier.predict_proba(X_no_df)
parent_dataset_no['Probability'] = list(proabability_prediction[:,0])
print(proabability_prediction[:0])
optimised_dataframe = parent_dataset_yes.append(parent_dataset_no)
optimised_dataframe.to_csv("optimised_dataframe.csv")
