import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

data_path = 'dataset.csv'

# Load the dataset using the load_data function
data_frame = pd.read_csv(data_path)

# Print the first five rows of the dataset
print(data_frame.head())

X = data_frame['Message'].values
y = data_frame['Category'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True)

vectorizer = TfidfVectorizer()
X_train=vectorizer.fit_transform(X_train)
X_test=vectorizer.transform(X_test)

svm=SVC()
svm.fit(X_train,y_train)
print(svm.score(X_test,y_test))
y_pred=svm.predict(X_test)
print(confusion_matrix(y_test,y_pred))