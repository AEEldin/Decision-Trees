# Decision-Trees
In this repository, we will discuss the work of decision trees algorithm using Python's Sklearn (scikit-learn).






## Step1: Prepare the required libraries

Python version 3.8 is the most stable version, used in the different tutorials. Scikit-learn (or commonly referred to as sklearn) is probably one of the most powerful and widely used Machine Learning libraries in Python.

```
python3 --version
pip3 --version
pip3 install --upgrade pip  
pip3 install pandas
pip3 install numpy
pip3 install scikit-learn
```



## Step2: Prepare/build your Dataset

```
from sklearn import tree
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
import matplotlib.pyplot as plt
```

Your dataset is composed of a set of records in a table format, and the titles of each column. Next, we build a dataframe out of the our dataset and display columns. A DataFrame is a 2D data structure (you can visualize it as a table or a spreadsheet) and is most commonly used pandas object. A DataFrame optionally accepts index (row labels) and columns (column lables) arguments. let's create a fake dataset as follows:

```
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
```


## Step3: Map some data points

Note, some ML, especially decision trees don't accept string values. We need to map them into integer values.
```
Month = {'March': 3, 'April': 4, 'April': 5}
Heat = {'Cold': 0, 'Hot': 1}
Rain = {'Yes':1, 'No':0}
dataSet['Month'] = dataSet['Month'].map(Month)
dataSet['Heat'] = dataSet['Heat'].map(Heat)
dataSet['Raining'] = dataSet['Raining'].map(Rain)

print(dataSet)
```

## Step4: Train the classifier (the model)
```
clf = tree.DecisionTreeClassifier()
# clf = tree.DecisionTreeClassifier(max_depth=3) #max_depth is maximum number of levels in the tree

clf.fit(dataSet,Labels)
```

## Step5: Predict outcome of new data records
use the trained model to predict the outcome of new datapoint(s)

```
Input = [[5, 0, 1]]       #Input = [['May', 'Cold', 'Yes']]
print(clf.predict(Input))
```

## Step6: Visualizing the trained tree
```
plt.figure(figsize=(25,10))
a = plot_tree(clf, 
              feature_names=titles, 
              class_names=Labels, 
              filled=True, 
              rounded=True, 
              fontsize=14)

plt.show()
```
