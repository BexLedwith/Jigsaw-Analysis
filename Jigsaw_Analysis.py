
#%%
# import libraries

from pprint import pprint
import re
from datetime import date, timedelta
import pandas as pd
import seaborn as sns
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import click
import sys

#%%
def load_data(file_name, date_column= 'created_date'):
    """read csv and make datetime column"""
    df = pd.read_csv("train.csv")
    df['date_only'] = pd.to_datetime(df['created_date']).dt.date
    print(df.shape)
    return df

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
    stopwords.update(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", 
        "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", 
        "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
        "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", 
        "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", 
        "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", 
        "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", 
        "against", "between", "into", "through", "during", "before", "after", "above", 
        "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", 
        "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", 
        "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", 
        "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", 
        "will", "just", "don", "should", "now", ' ', column])
    
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
    plt.savefig(f'{column} {date_}.png')
    


#%%
## peak_date function
def peak_date(df, column, offset):
    try:
        grouped = df.groupby('date_only')[column].sum().reset_index()
    except KeyError:
        print('This column does not exist:',column)
        sys.exit()
    return grouped.loc[grouped[column].idxmax()+offset]['date_only']

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



#%%
@click.command()
@click.argument('file')
@click.option('--column', default = 'female')
@click.option('--cutoff', default = 0.5)
@click.option('--max_words', default = 100)
@click.option('--n', default = 1)
@click.option('--offset', default=0)

def runner(file, column, cutoff, max_words, n, offset):
    df = load_data(file)
    wordcloud(df, column, peak_date(df, column, offset), cutoff, max_words = max_words, n = n)

if __name__ == '__main__':
    runner()

