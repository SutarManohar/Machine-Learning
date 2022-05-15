#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk


# In[2]:


nltk.download_shell()


# In[3]:


messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]


# In[4]:


print(len(messages))


# In[5]:


messages[50]


# In[7]:


for mess_no,message in enumerate(messages[:10]):
    print(mess_no,message)
    print('\n')


# In[8]:


messages[0]


# In[9]:


import pandas as pd


# In[10]:


messages = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',names = ['label','message'])


# In[11]:


messages.head()


# In[12]:


messages.describe()


# In[15]:


messages.groupby('label').describe()


# In[16]:


messages.groupby('label').describe().transpose()


# In[17]:


messages['length'] = messages['message'].apply(len)


# In[18]:


messages.head()


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[22]:


messages['length'].plot.hist(bins=150)


# In[23]:


messages['length'].describe()


# In[24]:


messages[messages['length']==910]


# In[25]:


messages[messages['length']==910]['message']


# In[26]:


messages[messages['length']==910]['message'].iloc[0]


# In[27]:


messages.hist(column='length',by='label',bins=60,figsize=(12,4))


# # TEXT PREPROCESSING

# In[28]:


import string


# In[29]:


mess = 'Sample message! Notice: it has punctuation.'


# In[30]:


string.punctuation


# In[31]:


nopunc  = [c for c in mess if c not in string.punctuation]


# In[32]:


nopunc


# In[34]:


from nltk.corpus import stopwords


# In[35]:


stopwords.words('english')


# In[39]:


nopunc = ''.join(nopunc)


# In[40]:


nopunc


# In[41]:


x = ['a', 'b', 'c', 'd']


# In[44]:


''.join(x)


# In[45]:


'&'.join(x)


# In[46]:


nopunc


# In[47]:


nopunc.split()


# In[49]:


clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[50]:


clean_mess


# In[ ]:


#def text_process(mess):
    '''
    1. remove punc
    2. remove stop words
    3. return list of clean text words
    '''


# In[56]:


def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    
    nopunc = ''.join(nopunc)
    
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[53]:


messages.head()


# In[54]:


messages['message'].head(5)


# In[57]:


messages['message'].head(5).apply(text_process)


# # Vectorization

# In[58]:


from sklearn.feature_extraction.text import CountVectorizer


# In[60]:


bow_transformer = CountVectorizer(analyzer = text_process).fit(messages['message'])


# In[63]:


bow_transformer


# In[65]:


print(len(bow_transformer.vocabulary_))


# In[67]:


mess4 = messages['message']


# In[68]:


mess4


# In[69]:


mess4 = messages['message'][3]


# In[70]:


mess4


# In[77]:


#bag of words - bow
bow4 = bow_transformer.transform([mess4])


# In[73]:


print(bow4)


# In[74]:


print(bow4.shape)


# In[75]:


bow_transformer.get_feature_names()[4068]


# In[76]:


bow_transformer.get_feature_names()[9554]


# In[78]:


messages_bow = bow_transformer.transform(messages['message'])


# In[99]:


print(messages_bow)


# In[80]:


print('Shape of Sprase Matrix: ',messages_bow.shape)


# In[101]:


print('Shape of Sprase Matrix: ',messages_bow.shape[0])


# In[102]:


print('Shape of Sprase Matrix: ',messages_bow.shape[1])


# In[100]:


print('Size of Sprase Matrix: ',messages_bow.size)


# In[81]:


#Non zero occurences

messages_bow.nnz


# In[ ]:


sparsity = (100.0* messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))


# In[103]:


print(sparsity)


# In[82]:


sparsity = (100.0* messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(round(sparsity)))


# In[84]:


sparsity = (100.0* messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format(sparsity))


# In[104]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[105]:


tfidf_transformer = TfidfTransformer().fit(messages_bow)


# In[107]:


tfidf4 = tfidf_transformer.transform(bow4)


# In[108]:


print(tfidf4)


# In[109]:


tfidf_transformer.idf_[bow_transformer.vocabulary_['university']]


# In[110]:


messages_tfidf = tfidf_transformer.transform(messages_bow)


# In[111]:


messages_tfidf


# In[112]:


from sklearn.naive_bayes import MultinomialNB


# In[113]:


spam_detect_model = MultinomialNB().fit(messages_tfidf,messages['label'])


# In[115]:


spam_detect_model.predict(tfidf4)[0]


# In[117]:


messages['label'][3]


# In[118]:


all_pred = spam_detect_model.predict(messages_tfidf)


# In[119]:


all_pred


# In[121]:


from sklearn.model_selection import train_test_split


# In[122]:


msg_train,msg_test,label_train,label_test = train_test_split(messages['message'],messages['label'],test_size=0.3)


# In[124]:


msg_train


# In[126]:


from sklearn.pipeline import Pipeline


# In[127]:


pipeline = Pipeline([('bow',CountVectorizer(analyzer=text_process)),('tfidf',TfidfTransformer()),
                    ('classifier',MultinomialNB())])


# In[128]:


pipeline.fit(msg_train,label_train)


# In[130]:


predictions = pipeline.predict(msg_test)


# In[131]:


from sklearn.metrics import classification_report


# In[132]:


print(classification_report(label_test,predictions))


# In[133]:


from sklearn.ensemble import RandomForestClassifier


# In[134]:


pipeline = Pipeline([('bow',CountVectorizer(analyzer=text_process)),('tfidf',TfidfTransformer()),
                    ('classifier',RandomForestClassifier())])


# In[135]:


pipeline.fit(msg_train,label_train)
predictions = pipeline.predict(msg_test)


# In[136]:


print(classification_report(label_test,predictions))

