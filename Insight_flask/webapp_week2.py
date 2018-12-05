from __future__ import print_function
import pickle
import pandas as pd
import numpy as np
import subprocess
import sklearn.datasets
from sklearn.decomposition import PCA
import argparse
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sklearn.ensemble
import lime
import lime.lime_tabular
from xgboost import XGBClassifier
from sklearn import cross_validation
import codecs
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import youtube_dl
import string
import glob
#From input url, extract data and views
DEVELOPER_KEY='your key'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
    developerKey=DEVELOPER_KEY)
def youtube_video(query):
  # Call the search.list method to retrieve results matching the specified
  # query term.
  search_response = youtube.videos().list(
    id=query,
    part='snippet,statistics'
  ).execute()

  videos = []

  # Add each result to the dataframe of videos
  global videos_df
  videos_ = []
  tags=[]
  
  for search_result in search_response.get('items', []):
      response2 = youtube.channels().list(
                  part='statistics, snippet',
                  id=search_result['snippet']['channelId']).execute()
      videos_.append({'VideoId': query, 'Title': search_result['snippet']['title'], 
                      'Description': search_result['snippet']['description'],
                      'channelID':search_result['snippet']['channelId'],
                      'channelSubscribers':response2['items'][0]['statistics']['subscriberCount'],
                      'publishedAt':search_result['snippet']['publishedAt'],
                     'viewCount':search_result['statistics']['viewCount']})
      if 'tags' in search_result['snippet'].keys():
            tags.append(search_result['snippet']['tags'])
      else:
            tags.append([])

  videos_df=pd.DataFrame(videos_)
  videos_df['tags']=tags

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def download_audio(videoid):
  ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'm4a',
    }],
  }
  
  with youtube_dl.YoutubeDL(ydl_opts) as ydl:
      ydl.download(videoid)



def get_dataset(input2):
  query=remove_prefix(input2,'https://www.youtube.com/watch?v=')
  query=query.split('&',1)[0] #NEW 10/2
  youtube_video(query)
  print(query)
  videos_df['Description']=videos_df['Description'].astype(str)
  videos_df['Title']=videos_df['Title'].astype(str)
  videos_df['viewCount']=videos_df['viewCount'].astype(float)
  videos_df['channelSubscribers']=videos_df['channelSubscribers'].astype(float)
  videos_df['publishedAt']=videos_df['publishedAt'].apply(lambda x: datetime.strptime(x,'%Y-%m-%dT%H:%M:%S.%fZ'))
  videos_df['Desc_word_count']=videos_df['Description'].str.split().str.len()
  a=datetime.now()
  videos_df['dayslive']=a-videos_df['publishedAt']
  videos_df['dayslive']=videos_df['dayslive'].apply(lambda x: (x.days * 86400 + x.seconds)/86400)
  title=videos_df['Title'][0]

  download_audio([input2])


  import moviepy.editor as mpy
  from pyAudioAnalysis import audioFeatureExtraction
  for file in glob.glob('*-{0}.m4a'.format(query)):
      clip=mpy.AudioFileClip(file)
  #clip = mpy.AudioFileClip('{0}-{1}.m4a'.format(title,query)) #Possible to modify duration of clip in this command
  sample_rate = clip.fps
  audio_data = clip.to_soundarray()
  print(len(audio_data))
  print(sample_rate)
  x1=[i[0] for i in audio_data]
  del audio_data
  del clip
  print("x1 complete")
  #x2=[i[1] for i in audio_data]
  F, f_names = audioFeatureExtraction.stFeatureExtraction(x1, sample_rate, 0.050*sample_rate, 0.025*sample_rate) #Not sure if this is the right frame size/overlap for classical music
  print("F exists")
  del x1
  sound_df=pd.DataFrame(data=F.transpose(),columns=f_names)
  sound_specs=sound_df.describe()
  del sound_df
  print("sound_specs exists")
  sound_specs = sound_specs.stack().to_frame().T
  sound_specs.columns = ['{}_{}'.format(*c) for c in sound_specs.columns]
  sound_specs = sound_specs.loc[:, ~sound_specs.columns.str.startswith('count_')]
  sound_specs['VideoId']=query #p  #NEW 10/2
  all_df=pd.merge(videos_df, sound_specs, on = 'VideoId')
  print("all_df exists")
  
  del sound_specs
  #Create text columns
  from nltk import word_tokenize, pos_tag, ne_chunk
  import nltk
  from collections import Counter
  #nltk.download('punkt')
  def count_ne(example):
        y=pos_tag(word_tokenize(example))
        counts = Counter(tag for word,tag in y)
        return counts['NNP']
  all_df['Description_ne_counts']= list(map(lambda x: count_ne(x), all_df['Description']))
  all_df['Title_ne_counts']= list(map(lambda x: count_ne(x), all_df['Title']))
  def count_ne_tags(example):
        ind=0
        for item in example:
            y=pos_tag(word_tokenize(item))
            counts = Counter(tag for word,tag in y)
            if counts['NNP']>0: 
                ind+=1
        return ind
  all_df['tags_ne_counts']= list(map(lambda x: count_ne_tags(x), all_df['tags']))
  translator = str.maketrans('', '', string.punctuation)
  all_df['tags'] = list(map(lambda x: [item.lower() for item in x], all_df['tags']))
  all_df['Title'] = list(map(lambda x: x.translate(translator).lower(), all_df['Title']))
  all_df['Description'] = list(map(lambda x: x.translate(translator).lower(), all_df['Description']))
  #remove special characters
  textrefs=pd.read_csv('Text lists.csv')
  composers=textrefs['Composer last name'].dropna()
  composers=list(map(lambda x: x.lower().rstrip(),composers))
  artists=textrefs['Famous violinists and pianists'].dropna()
  artists=list(map(lambda x: x.lower().rstrip(),artists))
  instruments=textrefs['Musical instrument'].dropna()
  instruments=list(map(lambda x: x.lower().rstrip(),instruments))
  genres=textrefs['Genre'].dropna()
  genres=list(map(lambda x: x.lower().rstrip(),genres))
  piece=textrefs['Piece type'].dropna()
  piece=list(map(lambda x: x.lower().rstrip(),piece))
  complete=textrefs['completeness'].dropna()
  complete=list(map(lambda x: x.lower().rstrip(),complete))
  def count_vars(example,elements):
        c=Counter(example.split())
        num=0
        for word in elements:
            num+=c[str(word)]
        return num
  def count_vars_tags(example,elements):
        c=Counter(example)
        num=0
        for word in elements:
            num+=c[str(word)]
        return num
    #count_vars(videodata['Description'][0],composers)
  all_df['tags_composer'] = list(map(lambda x: count_vars_tags(x,composers), all_df['tags']))
  all_df['Title_composer'] = list(map(lambda x: count_vars(x,composers), all_df['Title']))
  all_df['Description_composer'] = list(map(lambda x: count_vars(x,composers), all_df['Description']))
  all_df['tags_artists'] = list(map(lambda x: count_vars_tags(x,artists), all_df['tags']))
  all_df['Title_artists'] = list(map(lambda x: count_vars(x,artists), all_df['Title']))
  all_df['Description_artists'] = list(map(lambda x: count_vars(x,artists), all_df['Description']))
  all_df['tags_instruments'] = list(map(lambda x: count_vars_tags(x,instruments), all_df['tags']))
  all_df['Title_instruments'] = list(map(lambda x: count_vars(x,instruments), all_df['Title']))
  all_df['Description_instruments'] = list(map(lambda x: count_vars(x,instruments), all_df['Description']))
  all_df['tags_genres'] = list(map(lambda x: count_vars_tags(x,genres), all_df['tags']))
  all_df['Title_genres'] = list(map(lambda x: count_vars(x,genres), all_df['Title']))
  all_df['Description_genres'] = list(map(lambda x: count_vars(x,genres), all_df['Description']))
  all_df['tags_piece'] = list(map(lambda x: count_vars_tags(x,piece), all_df['tags']))
  all_df['Title_piece'] = list(map(lambda x: count_vars(x,piece), all_df['Title']))
  all_df['Description_piece'] = list(map(lambda x: count_vars(x,piece), all_df['Description']))
  all_df['tags_complete'] = list(map(lambda x: count_vars_tags(x,complete), all_df['tags']))
  all_df['Title_complete'] = list(map(lambda x: count_vars(x,complete), all_df['Title']))
  all_df['Description_complete'] = list(map(lambda x: count_vars(x,complete), all_df['Description']))
  all_df['tags_count']=list(map(lambda x: len(x), all_df['tags']))
  all_df['Title_length']=list(map(lambda x: len(x), all_df['Title']))
  from textblob import TextBlob
  all_df['Title_pos']=list(map(lambda x: TextBlob(x).sentiment[0], all_df['Title']))
  all_df['Description_pos']=list(map(lambda x: TextBlob(x).sentiment[0], all_df['Description']))
  all_df['tags_pos']=list(map(lambda x: TextBlob(" ".join(str(i) for i in x)).sentiment[0], all_df['tags']))

  #Project features to PCAs and convert to log for channelSubscribers, viewCount, dayslive
  all_data=all_df.drop(['publishedAt','tags','VideoId'], axis=1)
  #import saved PCA models
  pca = pickle.load(open('pca_model_mfcc.sav', 'rb'))
  X=all_data[['mean_mfcc_1','mean_mfcc_2','mean_mfcc_3','mean_mfcc_4','mean_mfcc_5','mean_mfcc_6','mean_mfcc_7','mean_mfcc_8','mean_mfcc_9','mean_mfcc_10','mean_mfcc_11','mean_mfcc_12','mean_mfcc_13','std_mfcc_1','std_mfcc_2','std_mfcc_3','std_mfcc_4','std_mfcc_5','std_mfcc_6','std_mfcc_7','std_mfcc_8','std_mfcc_9','std_mfcc_10','std_mfcc_11','std_mfcc_12','std_mfcc_13']]
  Xreg=pca.transform(X)
  print(Xreg)
  mfccdata=pd.DataFrame(Xreg,columns=['mfccPC1','mfccPC2','mfccPC3'])
  all_data=pd.concat([all_data,mfccdata],axis=1)
  #Create classifier column
  all_data['views_cat'] = np.where(all_data['viewCount']<200, 0,1)
  all_data.loc[all_data['viewCount']>=2000,'views_cat']=2
  all_data.loc[all_data['viewCount']>=20000,'views_cat']=3

  #Get rid of excess columns and reorder to match correct dataset
  all_data_reduced_final=all_data[['Desc_word_count','channelSubscribers','dayslive', 'mean_energy_entropy', 'mean_spectral_entropy',
       'Description_ne_counts', 'Title_ne_counts',
       'tags_composer', 'Title_composer', 'Description_composer',
       'tags_artists', 'Title_artists', 'Description_artists',
       'tags_instruments', 'Title_instruments', 'Description_instruments',
       'tags_genres', 'Title_genres', 'Description_genres', 'tags_count',
       'Title_pos', 'Description_pos', 'tags_pos', 'mfccPC1', 'mfccPC2','views_cat']]
  Y_test=all_data_reduced_final['views_cat']
  X_test=all_data_reduced_final.drop(['views_cat'],axis=1)
  #Create a bunch dataset
  testingdataset = sklearn.datasets.base.Bunch(data=X_test.values, target=Y_test.values, target_names=Y_test.name, feature_names=X_test.columns)
  return testingdataset

def get_results(data_bunch):
  #Run input data through model
  filename = 'final_xgb_model_week4.sav'
  loaded_model = pickle.load(open(filename, 'rb'))
  target_test=data_bunch.target
  test=data_bunch.data
  result2 = loaded_model.predict(test)
  return result2

def get_lime_results(data_bunch):
  #Run input data through model
  filename = 'final_xgb_model_week4.sav'
  loaded_model = pickle.load(open(filename, 'rb'))
  traindata=pd.read_csv('databasetraindata_4cat.csv')
  traindata=traindata.drop(['Unnamed: 0'],axis=1)
  Y_train=traindata['views_cat']
  X_train=traindata.drop(['views_cat','viewCount'],axis=1)
  trainingdataset = sklearn.datasets.base.Bunch(data=X_train.values, target=Y_train.values, target_names=Y_train.name, feature_names=X_train.columns)
  target_train=trainingdataset.target
  train=trainingdataset.data
  target_test=data_bunch.target
  test=data_bunch.data
  explainer = lime.lime_tabular.LimeTabularExplainer(train, feature_names=trainingdataset.feature_names, class_names=['Low','Medium','High','Viral'], discretize_continuous=False)
  exp = explainer.explain_instance(test[0], loaded_model.predict_proba, num_features=20, top_labels=4)
  return exp


def get_imp_table(data_bunch,exp,pred):
    filename = 'final_xgb_model_week4.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    # Create dataframe of feature ranking
    rankingtable=[]
    d_features={}
    d_features.update({'Desc_word_count':['Description length','Try writing more details in the description']})
    d_features.update({'tags_count':['Number of tags','Try adding relevant keywords to tags']})
    d_features.update({'mean_energy_entropy':['Mean energy entropy','Try normalizer or compression tool for volume issue or denoiser to address static or buzz']})
    d_features.update({'mean_spectral_entropy':['Mean spectral entropy','Adjust mic during recording to pick up different registers across instrument(s)']})
    d_features.update({'std_chroma_std':['Standard deviation of chroma features','Check for uneven playback speeds, out-of-tune playing, or excess noise']})
    d_features.update({'mfccPC1':['First PCA component of pitch and power spectra','Check recording for hum or low frequency noise, then apply denoiser tool or re-record after checking for wires for loops or voltage difference across electronics in recording set-up']})
    d_features.update({'mfccPC2':['Second PCA component of pitch and power spectra','Check recording for noise, buzz, clicks, or pops, then apply denoiser tool or re-record after checking environment for external noise, vibrations, or electronics affecting local wiring such as dimmer switch or motor']})
    d_features.update({'channelSubscribers':['Channel subscribers','Recruit subscribers']})
    d_features.update({'dayslive':['Length of time posted','Wait for more views']})
    d_features.update({'Description_ne_counts':['Desc ne cnts','Try adding named entities to the description']})
    d_features.update({'Title_ne_counts':['Title ne cnts','Try adding named entities to the title']})
    d_features.update({'tags_composer':['Composer tag','Include composer in video tags']})
    d_features.update({'Title_composer':['Composer title','Include composer in title']})
    d_features.update({'Description_composer':['Desc composer','Include composer in the description']})
    d_features.update({'tags_artists':['tag artists','Try adding artist names to the video tags']})
    d_features.update({'Title_artists':['title artists','Try adding artist names to the title']})
    d_features.update({'Description_artists':['desc artists','Try including artist names in description']})
    d_features.update({'tags_instruments':['tag instruments','Try including musical instruments in video tags']})
    d_features.update({'Title_instruments':['title instruments','Try including musical instruments in title']})
    d_features.update({'Description_instruments':['desc instruments','Try including musical instruments in description']})
    d_features.update({'tags_genres':['tag genre','Try adding musical genre to the video tags']})
    d_features.update({'Title_genres':['title genres','Try adding musical genre to the title']})
    d_features.update({'Description_genres':['desc genre','Try adding musical genre to the description']})
    d_features.update({'tags_count':['tag count','Try adding more keywords to the video tags']})
    d_features.update({'Title_pos':['title pos','Try including positive sentiment words in the title']})
    d_features.update({'Description_pos':['desc pos','Try including positive sentiment words in the description']})
    d_features.update({'tags_pos':['tags pos','Try including positive sentiment words in the video tags']})

    import emoji
    
    indices=data_bunch.feature_names.tolist()
    print(indices)

    print(pred)
    current=min(pred+1,3)
    print(current)
    if np.isscalar(current)==False:
        current=np.asscalar(current)
    explanations=exp.local_exp
    explanations=explanations[current]

    d_scores={}
    for i in range(len(explanations)):
      d_scores[explanations[i][0]]=explanations[i][1]
    for f in range(len(indices)):
        if f in d_scores:
            rankingtable.append({'Index':f,
                'Feature':d_features[indices[f]][0],
                'Current score':d_scores[f],
                'Suggestion':d_features[indices[f]][1],
                'Current performance':emoji.emojize(":smile:",use_aliases=True) if d_scores[f]>=0.025 else emoji.emojize(':rage:',use_aliases=True) if d_scores[f]<=-0.025 else emoji.emojize(':cold_sweat:',use_aliases=True) if d_scores[f]<=0 else emoji.emojize(':neutral_face:',use_aliases=True)})
        #sort ranking table so that worst scores are at the top
    rankingtable=pd.DataFrame(rankingtable)
    rankingtable=rankingtable.sort_values(by=['Current score'],ascending=True)
    rankingtable=rankingtable.loc[rankingtable['Current score']<-0.01]
    return rankingtable['Suggestion']
