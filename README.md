# topic_modeling
this is mainly for topic modeling based on users post.
Here is the raw code based on python

```
# this is for non negative matrix factorization
import pandas as pd
df = pd.read_csv("C:\\Users\\USER\\Desktop\\nlp_mini_prj_vds_csv_file.csv")
df['ID'] = [101,102,103,104,105,106,107,108,109,110]
df.set_index('ID',inplace=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
def topic_modelling(df):
    tfidf = TfidfVectorizer(max_df=1.0,min_df=1,stop_words='english')
    dtm = tfidf.fit_transform(df['Posts'])
    nmf_model = NMF(n_components=10,random_state=0)
    nmf_model.fit(dtm)
    for index,topic in enumerate(nmf_model.components_):
        print(f"The top 15 word for {index} is :")
        print([tfidf.get_feature_names()[i] for i in topic.argsort()[-20:]])
        print('\n')
    topic_res = nmf_model.transform(dtm)
    df['topic'] = topic_res.argmax(axis=1)
    my_topic_dict = {0:'LinkedIn',1:'ai course by laurence',2:'wroclaw-poland',3:'mandya farmers',
                4:'apple singapore',5:'campus education',6:'electric jeep',7:'telecom business dashboards using tablaeu',
                8:'automotive companies internship',9:'job support'}
    df['topic_labels'] = df['topic'].map(my_topic_dict)
    return df
topic_modelling(df)

#this code is for for latent dirichlets allocation

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
def dir_modeling(df):
  cv = CountVectorizer(max_df=1,min_df=1,stop_words='english')
  #dir_dtm = cv.fit_transform(df['Posts'])
  #dir_model = LatentDirichletAllocation(n_components=10,random_state=0)
  #dir_model.fit(dir_dtm)
  for index,topic in enumerate(dir_model.components_):
      print(f"the top 10 words for index {index} is :")
      print([cv.get_feature_names()[i] for i in topic.argsort()[-20:]])
      print('\n')
      print('\n')
  topic_results = dir_model.transform(dir_dtm)
  df['topic'] = topic_results.argmax(axis=1)
  my_topic_dict = {0:'LinkedIn',1:'ai course by laurence',2:'wroclaw-poland',3:'mandya farmers',
                  4:'apple singapore',5:'campus education',6:'electric jeep',7:'telecom business dashboards using tablaeu',
                  8:'automotive companies internship',9:'job support'}
  df['topic_labels'] = df['topic'].map(my_topic_dict)
  return df
dir_modeling(df)
topic_modelling(df)
```

# It is observed that , the dirichlets allocation is not so great as nmf when there was a comparision and hence, chose NMF for final topic modelling.
