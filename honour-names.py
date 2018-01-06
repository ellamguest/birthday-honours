#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:52:09 2018

@author: emg
"""


import requests
from bs4 import BeautifulSoup
import pandas as pd
from nameparser import HumanName
import numpy as np
import gender_guesser.detector as gender
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import spacy

nlp = spacy.load('en')

def get_csv_url(html):
    result = requests.get(html)
    if result.status_code == 404:
        result = requests.get(html.replace('list','lists'))
    soup = BeautifulSoup(result.content, 'html.parser')
    for link in soup.find_all('a'):
        if link['href'].endswith('csv'):
            return 'https://www.gov.uk' + link['href']
    else:
        return False

def check_headers(df, year):
    if 'Honours List {}'.format(year) in ' '.join(df.columns):
        df.columns = df.iloc[0]
        df = df[1:]
    df = df[['Order', 'Level', 'Award', 'Name', 'Citation', 'County']]
    return df

def prep_data(year):
    dfs = []
    
    nye_url = get_csv_url('https://www.gov.uk/government/publications/new-years-honours-list-{}'.format(year))
    if nye_url != False:
        nye_df = pd.read_csv(nye_url, encoding="ISO-8859-1", header=0)
        nye_df = check_headers(nye_df, year)
        nye_df['List'] = 'New Year'
        dfs.append(nye_df)

    birthday_url = get_csv_url('https://www.gov.uk/government/publications/birthday-honours-lists-{}'.format(year))
    if birthday_url != False:
        birthday_df = pd.read_csv(birthday_url, encoding="ISO-8859-1", header=0)
        birthday_df = check_headers(birthday_df, year)
        birthday_df['List'] = 'Birthday'
        dfs.append(birthday_df)
        
    df = pd.concat(dfs)
    df['Year'] = int(year)
    names = [name.replace(',', '') for name in df['Name']]
    hns = [HumanName(name) for name in names]
    df['title'] = [name.title for name in hns]
    df['first'] = [name.first for name in hns]
    
    return df

df = prep_data('2011')
df.head()

def compile_dataset():
    years = [year for year in range(2008,2019)]
    years_dfs = []
    for year in years:
        print(year)
        df = prep_data(year)
        years_dfs.append(df)
    full_df = pd.concat(years_dfs)
    
    full_df.to_csv('honours-2008-2018.csv')
    return full_df

def load():
    return pd.read_csv('honours-2008-2018.csv', index_col=0)

df = load()
df.head()

###### GET GENDERS
def get_award_gender(award):
    if 'Knight' in award:
        return 'male'
    if 'Dame' in award:
        return 'female'
    else:
        return 'andy'

def get_auto_genders():
    df = load()
    d = gender.Detector()
    df['name_gender'] = [d.get_gender(name) for name in df['first']]

    title_genders = {'Lord':'male', 'Lady':'female', 'Professor':'andy', 'Mr':'male',
     'Rt Hon':'andy', 'Dr':'andy', 'Baron':'male','Ms':'female', 'Sir':'male',
     'Miss':'female', 'Mrs':'female', 'Councillor':'andy', 'Lt Col':'andy', 
     'Reverend':'andy', 'Captain':'andy', 'Capt':'andy', 'Sister':'female',
     'The Reverend Deacon':'andy', '':'andy', 'Prof':'andy'}
    
    df['title_gender'] = df['title'].map(lambda x: title_genders.get(x, np.nan))
    
    df['award_gender'] = df['Award'].map(lambda x: get_award_gender(x))
    
    df['auto_gender'] = (df['award_gender'].map(lambda x: certain(x))
                         .combine_first(df['title_gender'].map(lambda x: certain(x)))
                         .combine_first(df['name_gender'].map(lambda x: certain(x)))
                         )
    
    return df

"""
GOT MANUAL GENDER CORRECTIONS FOR 2017 LIST - NOT PLANNING ON DOING FOR ALL
def add_manual_genders():
    df = get_auto_genders()
    '''up date dict for each dataset where no gender find by auto methods'''
    manual_gender = {
        'Dr Lesley SAWERS':'female',
        'Rt Hon Lindsay Harvey HOYLE MP':'male',
        'Carol LUKINS':'female',
        'Dr Tracey COOPER':'female',
        'Dr Tracey VELL':'female',
        'Professor Mary Josephine STOKES':'female',
        'Dr Kim Bernadette TAYLOR':'female',
        'Dr Lindsey Janet WHITEROD OBE':'female',
        'Dr Robin Howard LOVELL-BADGE FRS':'male',
        'Professor Charanjit BOUNTRA':'male',
        'Professor Sally-Ann COOPER':'female',
        'Professor Ngaire Tui WOODS': 'female',
        'Dr Siu Hung Robin SHAM':'male',
        'Dr Anwara ALI':'female',
        'Councillor Balwant Singh CHADHA':'male',
        'Lt Col (Retd) Mordaunt COHEN TD DL':'male',
        'Haji Mohammad Yaqub JOYA': 'male',
        'Very Rev Prof Iain Richard TORRANCE TD FRSE':'male',
        'Capt (rtd) Santa PUN':'unknown',
        'Dr Mehool Harshadray SANGHRAJKA':'male',
        'Dr Rohit SHANKAR':'male',
        'Dr Suryadevara Yadu  Porna Chandra Prasad RAO':'unknown'}

    df['manual_gender'] = df['Name'].map(lambda x: manual_gender.get(x.strip(), np.nan))
    
    return df

def select_genders():
    df = add_manual_genders()
    certain = lambda x: True if x == 'male' or x == 'female' else False
    df['final_gender'] = df['award_gender'].combine(df['title_gender'], lambda x1, x2: x1 if certain(x1) else x2)
    df['final_gender'] = df['final_gender'].combine(df['guess'], lambda x1, x2: x1 if certain(x1) else x2)
    df['final_gender'] = df['final_gender'].combine(df['manual_gender'], lambda x1, x2: x2 if certain(x2) else x1)
    
    df['count'] = 1
    
    return df

"""

df = get_auto_genders()

# COUNT BREAKDOWNS

cont = lambda x: df.pivot_table(index=x, columns='final_gender',values='count', aggfunc=sum, margins=True, fill_value=0)
cont('Year')
cont('Award')

def old_cont_plot(variable):
    table = cont(variable).drop('All')
    table.plot('female','male', kind='scatter', title = variable, c='r')
    
def cont_plot(variable):
    fig, ax = plt.subplots()
    table = cont(variable).drop('All')
    x,y = table['female'], table['male']
    
    
    if variable == 'Year':
        ax.scatter(x,y, alpha=0)
        years = table.index
        for i, year in enumerate(years):
            ax.annotate(str(int(year))[-2:], (x[i],y[i]))
    
    else:
        ax.scatter(x,y)
    
    ax.set_title('Female by Male Honours by {}'.format(variable))
    ax.set_xlabel('Number of Female Honorees')
    ax.set_ylabel('Number of Male Honorees')


cont_plot('Year')
cont_plot('Award')

compare = ['Award','County', 'Year', 'List']

for variable in compare:
    cont_plot(variable)
    
# title is interesting - seniority?, vestige of previous titles before quotas
# compare age to title seniority

c, p, dof, expected = chi2_contingency(cont('Level')[['female','male']])
c


# REASONS


doc = nlp(df['Citation'][0])
doc1 = nlp(df['Citation'][1])

docs = [nlp(doc) for doc in df['Citation']]

def get_noun_phrases(text):
    doc = nlp(str(text))
    noun_phrases = []
    for noun_phrase in doc.noun_chunks:
        noun_phrases.append(noun_phrase.text.lower())
    return noun_phrases

df['noun_phrases'] = df['Citation'].apply(lambda x: get_noun_phrases(x))

counts = cont('Level')[['male','female']].replace(0.0, np.nan).sort_values('male')
counts = counts.dropna(axis=0,how='any')
counts = counts.drop(['All'])

awards = counts.index

N = counts.shape[0]
men_means = np.log(counts['male'])

ind = np.arange(N)  # the x locations for the groups
width = 0.35  

fig, ax = plt.subplots()
rects1 = ax.bar(ind, men_means, width, color='b')

women_means = np.log(counts['female'])
rects2 = ax.bar(ind + width, women_means, width, color='r')

# add some text for labels, title and axes ticks
ax.set_ylabel('Log of Number of Awards')
ax.set_title('Number of Awards by Type and Gender')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(awards)

ax.legend((rects1[0], rects2[0]), ('Men', 'Women'))




certain = lambda x: x if x == 'male' or x == 'female' else np.nan
  df['final_gender'] = df['award_gender'].combine(df['title_gender'], 
    
    lambda x1, x2: x1 if certain(x1) else x2)

df1 = pd.DataFrame({'A': [0, 0], 'B': [4, 4]})
df2 = pd.DataFrame({'A': [1, 1], 'B': [3, 3]})
df1['A'].combine(df2['A'], lambda s1, s2: s1 if s1 < s2 else s2)

s1 = pd.Series([1, 2])
s2 = pd.Series([0, 3])
s1.combine(s2, lambda x1, x2: x1 if x1 < x2 else x2)

s3 = df['award_gender']
s4 = df['title_gender']

s3.combine(s4, lambda x1, x2: x1 if certain(x1) else x2)

gender1 = df['award_gender'].map(lambda x: certain(x))
gender2 = df['title_gender'].map(lambda x: certain(x))
gender3 = df['guess'].map(lambda x: certain(x))

final = (gender1
 .combine_first(gender2)
 .combine_first(gender3)
 )

final.value_counts()




final


