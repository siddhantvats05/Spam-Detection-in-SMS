import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import confusion_matrix

# Specify the path to the SPAM text message dataset
data_path = 'dataset.csv'

# Load the dataset using the load_data function
df= pd.read_csv(data_path)

# Print the first five rows of the dataset
print(df.head())

vectorizer=CountVectorizer(stop_words="english")
X=vectorizer.fit_transform(df['Message'])
y=df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True)

lr=LogisticRegression()
lr=lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)
print("Accuracy: ",accuracy_score(y_test,y_pred))

print("",confusion_matrix(y_test,y_pred))