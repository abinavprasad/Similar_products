import streamlit as st 
import pickle
import os
from io import BytesIO
import requests
import numpy as np
import tensorflow as tf
from annoy import AnnoyIndex
import pandas as pd
# from detectron2.change_bg import get_bg_changed
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
#import cv2
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
#from Preprocess import process_query_image_mc
#from models import load_model_from_db
#from load_feature_vec import load_features_from_db
#from grabcut import apply_grabcut
from ultralytics import YOLO
#from shapely.geometry import Polygon
import cv2
from Opensearchfn import get_similar_skus
from streamlit_image_select import image_select
from PIL import Image
from process_image import process_dataframe
from feature_extract import extract_features
from sklearn.metrics.pairwise import cosine_similarity
from st_clickable_images import clickable_images

def get_image(row):
    try:
        url = row['img']
        r = requests.get(url)
        img = Image.open(BytesIO(r.content))
        img_array = np.array(img)
        return url
    except:
        return "https://www.shutterstock.com/image-vector/default-avatar-profile-icon-social-media-1677509740"
def get_image_output(row):
    url = row.loc[0,'img']
    #st.write(url)
    r = requests.get(url)
    return BytesIO(r.content)
    #except:
    #    print("not_here")
    #    return "https://www.shutterstock.com/image-vector/default-avatar-profile-icon-social-media-1677509740"
    
def load_features_from_file(file_path):
    # Load image paths and features from a file using pickle
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data['features'],loaded_data['sku'], loaded_data['l1'],loaded_data['l2'], loaded_data['l3'], loaded_data['class']

def get_data(category,num):
    df=pd.read_csv('all_mens.csv')
    features,skus, l1,l2,l3,cat = load_features_from_file(f'Men_{category}.pkl')
    df_main=df[df['l2']==category]
    num=int(num)
    df=df_main.iloc[num:num+20]
    
    #df=df.head(20)
    return df_main,df,features,skus,l1,l2,l3,cat

def if_l3(features,skus,req_l3,l3_option):
    if l3_option=='No':
        return features,skus
    else:
        new_indices=[]
        for i in range(len(l3_db)):
            if l3_db[i] == rows['l3_actual'].iloc[0]:
                new_indices.append(i)
        new_features=[]
        new_skus=[]
        for i in new_indices:
            new_features.append(features_db[i])
            new_skus.append(skus_db[i])
    return new_features,new_skus




proceed=True
with st.sidebar:
    # st.[element_name]
    db = st.selectbox('local or elastic search',('Local','ES'))
    #l1 = st.selectbox('Choose l1 category',('Men', 'Women', 'Kids'))
    l1 = st.selectbox('Choose l1 category',('Men',))
    l2 = st.selectbox('Choose l2 category',('Clothing', 'Shoes', 'Bags'))
    l3_option= st.selectbox('Use l3 in search?',('Yes', 'No'))
    num = st.number_input("Insert a number")
    #l2 = st.selectbox('Choose l2 category',('Shoes',))
    #algo = st.selectbox('Choose algo',('yoloV1',))    
    if proceed:
        proceed=False
        images_list=[]
        skus_click=[]
        df_main,df,features_db,skus_db,l1_db,l2_db,l3_db,cat=get_data(l2,num)
        for i,row in df.iterrows():
            images_list.append(get_image(row))
            skus_click.append(row['sku'])
        
        #new_images=[]
        #for i in images_list:
        #    new_images.append(np.array(i))
        img = image_select(
        label="Select a sku",
        images=images_list,
        captions=skus_click,return_value="index",index= 0
        )
        url=images_list[int(str(img)[:100])]
        sku=skus_click[int(str(img)[:100])]
        rows=df.query(f'sku=="{sku}"')
        
    #div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    #img_style={"margin": "5px", "height": "200px"},
#)
    #st.write(str(img)[:100])
    #print(images_list)
    
    #st.write(rows)
#l3_db_upper = list(map(str.upper, l3_db))


features_db,skus_db=if_l3(features_db,skus_db,rows['l3_actual'].iloc[0],l3_option)

df['l3']=np.where(df['l2']=='Shoes','SHOE',df['l3'])
df['l3']=np.where((df['l2']=='Clothing') & df['l3'].isin(['SWEATERS & CARDIGANS','JACKETS & COATS','JACKETS']),'CLOTHING-LAYERED',df['l3'])
df['l3']=np.where((df['l2']=='Clothing') & df['l3'].isin(['TOPS','T-SHIRTS','SWEATERS & CARDIGANS','JACKETS & COATS','TOPWEAR','HOODIES & SWEATSHIRTS','SHIRTS','JACKETS','TOPS & T-SHIRTS','SWEATERS & KNITWEAR','POLO TEES','T-SHIRT']),'CLOTHING-TOP',df['l3'])
df['l3']=np.where((df['l2']=='Clothing') & df['l3'].isin(['SKIRTS','PANTS & LEGGINGS','JEANS','PANTS/CHINOS','PANTS & CHINOS','SHORTS & PANTS','PANTS','SHORTS','VEST & UNDERWEAR','INNER WEAR & THERMALS','SHORTS & PANTS','ATHLETIC WEAR']),'CLOTHING-BOTTOM',df['l3'])
df['l3']=np.where((df['l2']=='Clothing') & df['l3'].isin(['DRESSES','FROCKS & DRESSES','JUMPSUITS','MODEST WEAR','ARABIAN CLOTHING']),'CLOTHING-OVERALL',df['l3'])
df['l3']=np.where((df['l2']=='Bags') & df['l3'].isin(['HANDBAGS','HAND BAGS']),'BAGS-HANDBAG',df['l3'])
df['l3']=np.where((df['l2']=='Bags') & ~df['l3'].isin(['HANDBAGS','HAND BAGS']),'BAG',df['l3'])

classes={'CLOTHING-TOP': 0, 'CLOTHING-BOTTOM': 1, 'CLOTHING-OVERALL': 2, 'CLOTHING-LAYERED': 3, 'SHOE': 4, 'BAG': 5, 'BAG-HANDBAG': 6, 'CAP': 7, 'HAIR ACCESSORY': 8, 'SUNGLASS': 9, 'EARRING': 10, 'NECKLACE': 11, 'SCARF': 12, 'TIE': 13, 'RING': 14, 'WATCH': 15, 'GLOVE': 16, 'BELT': 17, 'SOCK': 18, 'SET': 19}

df=df.query('l3 in ("BAG","SHOE","CLOTHING-TOP","CLOTHING-OVERALL","CLOTHING-BOTTOM","CLOTHING-LAYERED")')
path=f"model_{l2}.h5"
#model=load_model(path)
#st.write(df[:10])
feature_vector=features_db[skus_db.index(rows['sku'].iloc[0])]
similarities = cosine_similarity(features_db)[skus_db.index(rows['sku'].iloc[0])]
print(similarities)
top_indices = np.argsort(similarities)[::-1][:20]
top_skus=[]
for i in top_indices:
    top_skus.append(skus_db[i])
#st.write(top_skus)
top_skus=tuple(top_skus)
df_final=df_main.query(f'sku in {top_skus}')
#st.write(feature_vector)
for i in top_skus[:20]:
        st.write(i)
        #st.image(get_image(d.query(f"sku=='{x}'").reset_index(drop=True)))
        st.image(get_image_output(df_final.query(f"sku=='{i}'").reset_index(drop=True)),width=100)
#l1=[]
#l2=[]
#l3=[]
#cat=[]
#for index,row in rows.iterrows():
#    yolo_output,clas=process_dataframe(row,classes)
#    features.append(extract_features(yolo_output, model))
#    l1.append(row['l1'])
#    l2.append(row['l2'])
#    l3.append(row['l3_actual'])
#    cat.append(row['l3'])






