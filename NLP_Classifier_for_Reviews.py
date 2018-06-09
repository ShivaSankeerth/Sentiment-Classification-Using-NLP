# Natural Language Processing based classifier for positive vs negative reviews

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('reviews_data.tsv', delimiter = '\t', quoting = 3)

print("NLP based classifier for Positive vs Negative reviews")

# Preprocessing the data
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 3000):
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

# Splitting the dataset into the Training and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print("******Predicting Review-type on Test Data****\nReview-number\tReview-type")
count=[0,0]                      #to count number of positive and negative reviews
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
plt.xlabel('Review Statements')
plt.ylabel('Positive or Negative Review')
plt.title("Review Classification into Positive or Negative Classes")

objects = ('Positive','Negative')
y_pos = np.arange(len(objects))

plt.barh(y_pos,count,align = 'center', alpha=0.5)
plt.yticks(y_pos, objects)
#plt.plot(list(y_pred))
plt.show()


print('confusion matrix\n',cm)
print('Accuracy Score:',acs*100,"%")
targetnames=["Negative","Positive"]
print("Classification Report:\n",classification_report(y_test,y_pred,target_names=targetnames ))

'''
References:
Udemy course(Kirill Eremenko's)
Machine Learning articles on Medium.com
Documentations on python.org,matplotlib.org,etc.
LinkedIn articles
'''
