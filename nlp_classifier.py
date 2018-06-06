# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("******Predicting Review-type on Test Data****\nReview-number\tReview-type")
count=[0,0]
for index,i in enumerate(y_pred):
    if(i==1):
        print(index+1,"\t\tPositive review :)")
        count[0]=count[0]+1
    else:
        print(index+1,"\t\tNegative Review :(")
        count[1]=count[1]+1
print("Positive Reviews:",count[0],"\nNegative Reviews:",count[1])
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
acs=accuracy_score(y_test,y_pred)
plt.xlabel('Reviews')
plt.ylabel('Positive or Negative')
plt.title("Review Classification into Positive or Negative Classes")
plt.plot(list(y_pred))
plt.show()
print('confusion matrix\n',cm)
print('Accuracy Score:',acs*100,"percent")
targetnames=["negative","positive"]
print("Classification Report:\n",classification_report(y_test,y_pred,target_names=targetnames ))
