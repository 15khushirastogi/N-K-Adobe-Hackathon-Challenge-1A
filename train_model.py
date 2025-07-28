import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

path = 'training_csvs'
files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]

df_list = []
for f in files:
    try:
        df = pd.read_csv(f, encoding='ISO-8859-1') 
        df_list.append(df)
    except Exception as e:
        print(f"❌ Failed to read {f}: {e}")

df = pd.concat(df_list, ignore_index=True)

df = df.dropna(subset=['label'])  

X = df['text'].astype(str)       
y = df['label'].astype(str)       


le = LabelEncoder()
y_encoded = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)


print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, labels=np.arange(len(le.classes_)), target_names=le.classes_))


with open('text_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print(" Model, vectorizer, and label encoder saved.")