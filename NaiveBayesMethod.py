import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data_path = 'dataset.csv'
data_frame = pd.read_csv(data_path)
print(data_frame.head())

# Extract X and y from the data frame
X = data_frame['Message'].tolist()
y = data_frame['Category'].tolist()

# Initialize label encoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Print the first 5 elements of X and y
print(f'X[:5]: \n{X[:5]}\n')
print(f'y[:5]: {y[:5]}\n')
print(f"Label Mapping : {label_encoder.inverse_transform(y[:5])}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True)
clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])
clf.fit(X_train,y_train)
print("Accuracy: ",clf.score(X_test,y_test))
y_pred=clf.predict(X_test)
print("",confusion_matrix(y_test,y_pred))