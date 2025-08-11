import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Loads the penguin dataset from a CSV file into a pandas DataFrame
penguin_df = pd.read_csv('penguins.csv')

# Removes all rows containing missing values (NaN) from the dataset
penguin_df.dropna(inplace=True)  # inplace=True modifies the original DataFrame rather than creating a new one


output = penguin_df['species']  # Extracts the 'species' column as the target variable (what we want to predict)

# Selects the feature columns (input variables) that will be used to predict species
features = penguin_df[['island',  'bill_length_mm', 'bill_depth_mm',
                       'flipper_length_mm', 'body_mass_g', 'sex']]

features = pd.get_dummies(features)  # Converts categorical variables (island, sex) into numerical format using one-hot encoding

# Converts species names to numerical labels (0, 1, 2)
output, uniques = pd.factorize(output) # output becomes an array of integers representing each species
                                        # uniques stores the original species names for later reference

# x_train, y_train: Training features and label
# x_test, y_test: Testing features and labels
x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=.8)

# Creates a Random Forest classifier with a fixed random seed for reproducible results
rfc = RandomForestClassifier(random_state=15)

# Trains the model using training data
rfc.fit(x_train.values, y_train)  # .values converts pandas DataFrame to numpy array (required by scikit-learn)

# Makes predictions on the test set using the trained model
y_pred = rfc.predict(x_test)

# Calculates accuracy by comparing predictions to actual test labels
score = accuracy_score(y_pred, y_test)
print('Our accuracy score for this model is {}'.format(score))  # Prints the accuracy score (percentage of correct predictions)

# Saves the trained model to a file for future use
rf_pickle = open('random_forest_penguin.pickle', 'wb')
pickle.dump(rfc, rf_pickle)
rf_pickle.close()

# Saves the species name mappings (from pd.factorize) to a separate file
output_pickle = open('output_penguin.pickle', 'wb')
pickle.dump(uniques, output_pickle)
output_pickle.close()
fig, ax = plt.subplots()
ax = sns.barplot(x=rfc.feature_importances_, y=features.columns)
plt.title('Which features are the most important for species prediction?')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
fig.savefig('feature_importance.png')



