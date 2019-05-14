
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
## string counts function
def str_counts(s: str, target: str) -> int:
    return len(re.findall(target, s))

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




