
#%%
# import libraries

from pprint import pprint
import re
from datetime import date
import pandas as pd
import seaborn as sns
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#%%
# read csv
df = pd.read_csv("train.csv")



#%%
# binarize threat and obscene
max_threat = df['threat'] >= 0.5
df['threat_binary'] = max_threat
df['threat_binary'].astype(int)

max_obscene = df['obscene'] >= 0.5
df['obscene_binary'] = max_obscene
df['obscene_binary'].astype(int)


#%%
# make date column
df['date_only'] = pd.to_datetime(df['created_date']).dt.date


#%% 
# counts_over_time
counts_over_time = df.groupby('date_only')['id'].count().reset_index()

#%%
# all_ethnicities
all_ethnicities = df.groupby('date_only')['asian', 'black', 'white', 'latino', 'other_race_or_ethnicity'].sum().reset_index()

#%%
# melted_ethnicities
melted_ethnicities = all_ethnicities.melt('date_only', var_name = 'Ethnicities', value_name = 'count')


#%%
## peak_white (charlottesville)
peak_white_mask = (df['date_only'] == date(2017,8,16)) & (df['white'] >= 0.5)
charl = df[peak_white_mask]

#%%
## fake_news_mask
fake_news_mask = df['comment_text'].str.lower().str.contains('fake news')
df[fake_news_mask]


#%%
## string counts function
def str_counts(s: str, target: str) -> int:
    return len(re.findall(target, s))


#%%
# fake news in comments
df['comment_text'].apply(str_counts, args = ('fake news',)).idxmax()
df.loc[384103]['comment_text']

#%%
mask = df.groupby('article_id')['comment_text']


#%% [markdown]
# mask.apply(str_counts, args = ('fake news',)).idxmax()


#%%
## peak_asian 
df.loc[df['asian'].idxmax()]
peak_asian_mask = (df['date_only'] == date(2017,7,14)) & (df['asian'] >= 0.5)
bastille_day = df[peak_asian_mask]


#%%
## day before peak asian
pre_bastille_mask = (df['date_only'] == date(2017,7,13)) & (df['asian'] >= 0.5)
pre_bastille = df[pre_bastille_mask]


#%%
## peak black
all_ethnicities.loc[all_ethnicities['black'].idxmax()]
peak_black_mask = (df['date_only'] == date(2017,9,26)) & (df['black'] >= 0.5)
kneeling = df[peak_black_mask]

#%%
# threats
threat_mask = df['threat_binary'] == 1
threats = df[threat_mask]

#%%
## non_threats
non_threat_mask = df['threat_binary'] == 0
non_threat = df[non_threat_mask]

#%%
## obscene
obscene_mask = df['obscene_binary'] == 1
obscene = df[obscene_mask]


#%%
## sex and gender
sex_gender = df.groupby('date_only')['male', 'female', 'other_gender', 'transgender'].sum().reset_index()
melted_sex_gender = sex_gender.melt('date_only', var_name = 'Sex_or_Gender', value_name = 'count')


#%%
## peak female
sex_gender.loc[sex_gender['female'].idxmax()]['date_only']
peak_female_mask = (df['date_only'] == date(2017,1,22)) & (df['female'] >= 0.5)
womens_march = df[peak_female_mask]

#%%
## article comments
article_comments = df.groupby('article_id')['article_id','comment_text'].apply(lambda x : x)
article_comments['comment_text'].apply(str_counts, args = ('fake news',)).idxmax()


#%%
## peak fake news
article_comments.loc[384103]['comment_text']

#%%
## my wordcloud function
def wordcloud(df, column, date_, cutoff = 0.5, **kwargs):
    mask = (df['date_only'] == date_) & (df[column] >= cutoff)
    new_mask = df[mask]
    
    stopwords = set(STOPWORDS)
    stopwords.update(["this", "the", "that", "with", ' ', column])
    
    text = " ".join(x for x in new_mask['comment_text'])
    
    text = re.sub(r'\W',' ', text)
    text = text.replace('\n', ' ')
    words = (word.strip() for word in text.split(' ') if word.lower() not in stopwords)
    big_str = ' '.join(words)
    
    if 'max_words' not in kwargs:
        max_words = 20
    else:
        max_words = kwargs['max_words']

#     max_words = kwargs.get('max_words', 20)
    d = ngrams(big_str, kwargs['n'])    
    wordcloud = WordCloud(stopwords=stopwords, max_words= max_words, 
                          background_color="white", width=2000, height=2000).generate_from_frequencies(d)
    
    plt.figure(figsize=(12,12))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
    pprint(d)


#%%
## peak_date function
def peak_date(df, column):
    grouped = df.groupby('date_only')[column].sum().reset_index()
    return grouped.loc[grouped[column].idxmax()]['date_only']

#%%
## ngrams function
def ngrams(input, n):
    input = input.split(' ')
    output = {}
    for i in range(len(input)-n+1):
        g = ' '.join(input[i:i+n])
        output.setdefault(g, 0)
        output[g] += 1
    return output




