from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
import matplotlib.pyplot as plt



# let's create a fake dataset as follows:

# data enties (called input or features) are Month, Heat, Raining Condition
titles = ['Month','Heat','Raining']
features = [
    ['March', 'Cold', 'No'],
    ['March', 'Cold', 'Yes'], 
    ['April', 'Hot', 'No'],
    ['April', 'Hot', 'Yes']
]
dataSet = pd.DataFrame(features,columns=titles)

# decision for each data entry (called outcome or label)
Labels = ['No Umberlla', 'Umberlla', 'No Umberlla', 'Umberlla']




# Note, some ML, especially decision trees don't accept string values
# We need to map them into integer values

Month = {'March': 3, 'April': 4, 'April': 5}
Heat = {'Cold': 0, 'Hot': 1}
Rain = {'Yes':1, 'No':0}
dataSet['Month'] = dataSet['Month'].map(Month)
dataSet['Heat'] = dataSet['Heat'].map(Heat)
dataSet['Raining'] = dataSet['Raining'].map(Rain)

print(dataSet)



# train the classifier (the model)

clf = tree.DecisionTreeClassifier()
# clf = tree.DecisionTreeClassifier(max_depth=3) #max_depth is maximum number of levels in the tree

clf.fit(dataSet,Labels)



# use the trained model to predict the outcome of new datapoint(s)

Input = [[5, 0, 1]]       #Input = [['May', 'Cold', 'Yes']]

print(clf.predict(Input))




# visualizing the tree
plt.figure(figsize=(25,10))
a = plot_tree(clf, 
              feature_names=titles, 
              class_names=Labels, 
              filled=True, 
              rounded=True, 
              fontsize=14)

plt.show()
