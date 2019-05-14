
#%%
import pandas as pd
import seaborn as sns
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from pprint import pprint


#%%
df = pd.read_csv("train.csv")


#%%
df.head()


#%%
df.shape


#%%
df.describe()


#%%
df["threat"]


#%%
pos_threat = df['threat'] == 1
obscene = df['obscene'] == 1


#%%
obscene.sum()


#%%
df.loc[(pos_threat) & (~obscene), ['threat', 'obscene']].count()


#%%
df.loc[(obscene) & (~pos_threat), ['threat', 'obscene']].count()


#%%
df.loc[(~obscene) & (~pos_threat), ['threat', 'obscene' ]].count()


#%%
df.groupby('threat')['obscene'].count()


#%%
max_threat = df['threat'] >= 0.5
df['threat_binary'] = max_threat


#%%
df['threat_binary'].astype(int)


#%%
max_obscene = df['obscene'] >= 0.5
df['obscene_binary'] = max_obscene
df['obscene_binary'].astype(int)


#%%
df.groupby('threat_binary')['obscene_binary'].count()


#%%
df.groupby(['threat_binary', 'obscene_binary'])['id'].count()


#%%
df.head(2)


#%%
df.columns


#%%
df['date_only'] = pd.to_datetime(df['created_date']).dt.date


#%%
counts_over_time = df.groupby('date_only')['id'].count().reset_index()


#%%
sns.set(rc={'figure.figsize': (11.7,8.27)})


#%%
sns.lineplot(x = 'date_only', y = 'id', data = counts_over_time)


#%%
all_ethnicities = df.groupby('date_only')['asian', 'black', 'white', 'latino', 'other_race_or_ethnicity'].sum().reset_index()


#%%
all_ethnicities


#%%
melted_ethnicities = all_ethnicities.melt('date_only', var_name = 'Ethnicities', value_name = 'count')


#%%
sns.set(rc={'figure.figsize': (11.7,8.27)})


#%%
sns.lineplot(x= 'date_only', y = 'count', data = melted_ethnicities, hue = 'Ethnicities')


#%%
all_ethnicities.loc[all_ethnicities['white'].idxmax()]


#%%
melted_ethnicities.head(15)


#%%
all_ethnicities.head()


#%%
all_ethnicities.plot()


#%%
from datetime import date
peak_white_mask = (df['date_only'] == date(2017,8,16)) & (df['white'] >= 0.5)


#%%
peak_white_mask.sum()


#%%
charl = df[peak_white_mask]


#%%
charl['comment_text'].values


#%%
for x in charl['comment_text']:
    print(x)


#%%
get_ipython().run_cell_magic('timeit', '', "counter = 0\nfor x in df['comment_text']:\n    if 'fake news' in x.lower():\n        counter += 1")


#%%
get_ipython().run_cell_magic('timeit', '', "l = [x for x in df['comment_text'] if 'fake news' in x.lower()]")


#%%
mask = df['comment_text'].str.lower().str.contains('fake news')


#%%
df[mask]


#%%
import re 


#%%
def str_counts(s: str, target: str) -> int:
    return len(re.findall(target, s))


#%%
s = "This fake news is such fake news"
str_counts(s,'fake news')


#%%
df['comment_text'].apply(str_counts, args = ('fake news',)).idxmax()


#%%
df.loc[384103]['comment_text']


#%%
df['mask'] = df['comment_text']


#%%
mask = df.groupby('article_id')['mask']


#%%
df['comment_text'].describe()

#%% [markdown]
# mask.apply(str_counts, args = ('fake news',)).idxmax()

#%%
get_ipython().run_line_magic('pinfo', 'WordCloud')


#%%
text = " ".join(x for x in charl['comment_text'])
stopwords = set(STOPWORDS)
stopwords.update(["this", "the", "that", "with", "white"])


#%%
wordcloud = WordCloud(stopwords=stopwords, max_words=100, background_color="white").generate(text)


#%%
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#%%
df.loc[df['asian'].idxmax()]


#%%
peak_asian_mask = (df['date_only'] == date(2017,7,14)) & (df['asian'] >= 0.5)
bastille_day = df[peak_asian_mask]


#%%
for x in bastille_day['comment_text']:
    print(x)


#%%
text = " ".join(x for x in bastille_day['comment_text'])
stopwords = set(STOPWORDS)
stopwords.update(["this", "the", "that", "with", "asian"])


#%%
wordcloud = WordCloud(stopwords=stopwords, max_words=100, background_color="white").generate(text)


#%%
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#%%
pre_bastille_mask = (df['date_only'] == date(2017,7,13)) & (df['asian'] >= 0.5)
pre_bastille = df[pre_bastille_mask]


#%%
text = " ".join(x for x in pre_bastille['comment_text'])
stopwords = set(STOPWORDS)
stopwords.update(["this", "the", "that", "with", "asian"])


#%%
wordcloud = WordCloud(stopwords=stopwords, max_words=200, background_color="white").generate(text)


#%%
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#%%
all_ethnicities.loc[all_ethnicities['black'].idxmax()]


#%%
peak_black_mask = (df['date_only'] == date(2017,9,26)) & (df['black'] >= 0.5)
kneeling = df[peak_black_mask]


#%%
for x in kneeling['comment_text']:
    print(x)


#%%
text = " ".join(x for x in kneeling['comment_text'])
stopwords = set(STOPWORDS)
stopwords.update(["this", "the", "that", "with", "black"])


#%%
wordcloud = WordCloud(stopwords=stopwords, max_words=300, background_color="white").generate(text)


#%%
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#%%
threat_mask = df['threat_binary'] == 1
threats = df[threat_mask]


#%%
text = " ".join(x for x in threats['comment_text'])
stopwords = set(STOPWORDS)
stopwords.update(["this", "the", "that", "with"])


#%%
wordcloud = WordCloud(stopwords=stopwords, max_words=100, background_color="white").generate(text)


#%%
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#%%
non_threat_mask = df['threat_binary'] == 0
non_threat = df[non_threat_mask]


#%%
text = " ".join(x for x in non_threat['comment_text'])
stopwords = set(STOPWORDS)
stopwords.update(["this", "the", "that", "with"])


#%%
wordcloud = WordCloud(stopwords=stopwords, max_words=100, background_color="white").generate(text)


#%%
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#%%
obscene_mask = df['obscene_binary'] == 1
obscene = df[obscene_mask]


#%%
text = " ".join(x for x in obscene['comment_text'])
stopwords = set(STOPWORDS)
stopwords.update(["this", "the", "that", "with"])


#%%
wordcloud = WordCloud(stopwords=stopwords, max_words=100, background_color="white").generate(text)


#%%
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#%%
sex_gender = df.groupby('date_only')['male', 'female', 'other_gender', 'transgender'].sum().reset_index()


#%%
sex_gender


#%%
melted_sex_gender = sex_gender.melt('date_only', var_name = 'Sex_or_Gender', value_name = 'count')


#%%
sns.set(rc={'figure.figsize': (11.7,8.27)})


#%%
sns.lineplot(x= 'date_only', y = 'count', data = melted_sex_gender, hue = 'Sex_or_Gender')


#%%
sex_gender.loc[sex_gender['female'].idxmax()]['date_only']


#%%
peak_female_mask = (df['date_only'] == date(2017,1,22)) & (df['female'] >= 0.5)
womens_march = df[peak_female_mask]


#%%
for x in womens_march['comment_text']:
    print(x)


#%%
text = " ".join(x for x in womens_march['comment_text'])
stopwords = set(STOPWORDS)
stopwords.update(["this", "the", "that", "with"])


#%%
wordcloud = WordCloud(stopwords=stopwords, max_words=300, background_color="white").generate(text)


#%%
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


#%%
df['article_id']


#%%
article_comments = df.groupby('article_id')['article_id','comment_text'].apply(lambda x : x)


#%%
print(article_comments)


#%%
article_comments['comment_text'].apply(str_counts, args = ('fake news',)).idxmax()


#%%
article_comments.loc[384103]['comment_text']


#%%
get_ipython().run_line_magic('pinfo', 'WordCloud')


#%%
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
wordcloud(df, 'asian', date

(TD also gives out really spiffy calendars in Chinese, while the rest of us receive a more ordinary version.)
Newsflash, Globe! People did in custody in China as Harper signed his deal with China and allowed it to buy the BC coal mine that it staffed with Chinese nationals.
"That is called rule of law, blame yourself.

I am not going to invest anything owned by Hongkong guys. They have nothing besides mouth water. Left hand borrow, right hand sell, which is their business practice, nothing from themselves."
.
.
.
Hong Kong is part of China. It's citizens are Chinese.

Case closed.(2017,7,14), max_words = 100) 


#%%
def peak_date(df, column):
    grouped = df.groupby('date_only')[column].sum().reset_index()
    return grouped.loc[grouped[column].idxmax()]['date_only']


#%%
peak_date(df, 'female')


#%%
wordcloud(df, 'female', peak_date(df, 'female'), max_words= 200)


#%%
for race in ['asian', 'black', 'white', 'latino', 'other_race_or_ethnicity']:
    print(race, peak_date(df, race))
    wordcloud(df, race, peak_date(df, race))


#%%
def ngrams(input, n):
    input = input.split(' ')
    output = {}
    for i in range(len(input)-n+1):
        g = ' '.join(input[i:i+n])
        output.setdefault(g, 0)
        output[g] += 1
    return output


#%%
ngrams('a a a b b c c', 2)


#%%
# d = ngrams(' '.join(x for x in womens_march['comment_text']), 2)
wordcloud(df, 'female', peak_date(df, 'female'), max_words= 100, n=2)


#%%
ngrams(' i am i     i i ', 2)


#%%



