#!/usr/bin/env python
# coding: utf-8

# ## Fake News Detection

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[10]:


true_df = pd.read_csv('True.csv')
fake_df = pd.read_csv('Fake.csv')


# In[11]:


true_df['label'] = 0


# In[12]:


fake_df['label'] = 1


# In[13]:


true_df.head()


# In[14]:


fake_df.head()


# In[15]:


true_df = true_df[['text','label']]
fake_df = fake_df[['text','label']]


# In[16]:


dataset = pd.concat([true_df , fake_df])


# In[17]:


dataset.shape


# ### Null values

# In[18]:


dataset.isnull().sum() # no null values


# ### Balanced or Unbalanced dataset

# In[19]:


dataset['label'].value_counts()


# In[20]:


true_df.shape # true news


# In[21]:


fake_df.shape # fake news


# ### Shuffle or Resample

# In[22]:


dataset = dataset.sample(frac = 1)


# In[23]:


dataset.head(20)


# In[31]:


import nltk

nltk.download('stopwords')


# In[32]:


import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[33]:


lemmatizer = WordNetLemmatizer()


# In[34]:


stopwords = stopwords.words('english')


# In[35]:


nltk.download('wordnet')


# In[46]:


import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_data(text):
    text = text.lower()  # convert to lowercase
    text = re.sub('[^a-zA-Z]', ' ', text)  # remove non-alphabets
    tokens = text.split()  # split into words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]  
    clean_text = ' '.join(tokens)  # join back into a string
    return clean_text


# In[47]:


dataset['text'] = dataset['text'].astype(str).apply(clean_data)


# In[48]:


dataset.isnull().sum()


# In[49]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[50]:


vectorizer = TfidfVectorizer(max_features = 50000 , lowercase=False , ngram_range=(1,2))


# In[51]:


X = dataset.iloc[:35000,0]
y = dataset.iloc[:35000,1]


# In[52]:


X.head()


# In[56]:


y.head()


# In[70]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Features (text) and labels
X = dataset['text']
y = dataset['label']   # change to your actual target column name

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000)
vec_train = vectorizer.fit_transform(X_train)
vec_test = vectorizer.transform(X_test)

# ✅ shapes match now
print(vec_train.shape, len(y_train))
print(vec_test.shape, len(y_test))


# In[71]:


clf.fit(vec_train, y_train)
predictions = clf.predict(vec_test)


# In[77]:


vec_train = vec_train.toarray()


# In[78]:


vec_test = vectorizer.transform(test_X).toarray()


# In[79]:


train_data = pd.DataFrame(vec_train, columns=vectorizer.get_feature_names_out())
test_data = pd.DataFrame(vec_test, columns=vectorizer.get_feature_names_out())


# ## Multinomial NB

# In[80]:


from sklearn.naive_bayes import MultinomialNB


# In[81]:


from sklearn.metrics import accuracy_score,classification_report


# In[82]:


clf = MultinomialNB()


# In[83]:


clf.fit(train_data, train_y)
predictions  = clf.predict(test_data)


# In[85]:


print(classification_report(test_y , predictions))


# In[86]:


import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Text Cleaning Function
# -----------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_data(text):
    if pd.isnull(text):   # handle missing values
        return ""
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# -----------------------------
# 2. Apply Cleaning
# -----------------------------
dataset['text'] = dataset['text'].astype(str).apply(clean_data)

# -----------------------------
# 3. Split Data
# -----------------------------
X = dataset['text']
y = dataset['label']   # ⚠️ change 'label' to your actual target column

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Vectorization
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
vec_train = vectorizer.fit_transform(X_train)
vec_test = vectorizer.transform(X_test)

# -----------------------------
# 5. Train Model
# -----------------------------
clf = MultinomialNB()
clf.fit(vec_train, y_train)

# -----------------------------
# 6. Evaluate
# -----------------------------
predictions = clf.predict(vec_test)

print("✅ Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))


# Now predict on both train set

# In[88]:


# Predict on training data
predictions_train = clf.predict(vec_train)

# Now you can calculate accuracy
from sklearn.metrics import accuracy_score

train_accuracy = accuracy_score(y_train, predictions_train)
test_accuracy = accuracy_score(y_test, predictions)

print("Training Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)


# In[89]:


accuracy_score(train_y , predictions_train)


# In[90]:


accuracy_score(test_y , predictions)


# In[ ]:




