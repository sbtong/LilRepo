import macpy.utils.database as db
import pandas as pd
from collections import defaultdict
import numpy as np
import json

from sklearn.feature_extraction import DictVectorizer

def onehot_encode(df, cols):
    vec = DictVectorizer()
    
    vec_data = pd.DataFrame(vec.fit_transform(df[cols].to_dict(outtype='records')).toarray())
    vec_data.columns = vec.get_feature_names()
    vec_data.index = df.index
    
    df = df.drop(cols, axis=1)
    df = df.join(vec_data)
    return df

def integer_encode(df, col_name):
    a = []
    ids = df[col_name].unique()
    for x in df[col_name].values:
        a.append(np.where(ids == x)[0][0])
    return a

def fdrec(df):
    drec = dict(df.irow(0).dropna())
    drec = process_frame(df, drec)
    return drec

def process_frame(df, drec, level=0):
    children = []
    tdf = df[df['ParentIssuerId'] == drec['IssuerId']]
    if tdf.empty:
        return drec
    if tdf.shape[0] == 1:
        drec['children'] = [tdf.irow(0).dropna().to_dict()]
        return drec
    for i, x in tdf.iterrows():
        d = x.replace('', np.nan).dropna().to_dict()
        children.append(d)
    drec["children"] = children
    for child in children:
        process_frame(df, child, level=level+1)
    return drec

def load_data(issuer_id=None, ISIN=None):
    if issuer_id:
        dbconn = db.RiskEntity(issuer_id=issuer_id)
    if ISIN:
        dbconn = db.RiskEntity(ISIN=ISIN)
    df = dbconn.extract_from_db()
    df.rename(columns={"Moody's":'Moodys'}, inplace=True)
    return df

def run(issuer_id=None, ISIN=None):
    if issuer_id:
        df = load_data(issuer_id=issuer_id)
    if ISIN:
        df = load_data(ISIN=ISIN)
    if df.empty:
        return None
    df['RiskEntityIdInt'] = onehot_encode(df,['RiskEntityId'])
    js = json.dumps(fdrec(df))
    return js