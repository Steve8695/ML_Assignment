from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


app = Flask(__name__, template_folder='template')
model = pickle.load(open('best_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("homepage.html")
def get_data():
    CR_PROD_CNT_IL = request.form.get('CR_PROD_CNT_IL')
    AMOUNT_RUB_CLO_PRC = request.form.get('AMOUNT_RUB_CLO_PRC')
    PRC_ACCEPTS_A_EMAIL_LINK = request.form.get('PRC_ACCEPTS_A_EMAIL_LINK')
    APP_REGISTR_RGN_CODE = request.form.get('APP_REGISTR_RGN_CODE')
    PRC_ACCEPTS_A_POS = request.form.get('PRC_ACCEPTS_A_POS')
    PRC_ACCEPTS_A_TK = request.form.get('PRC_ACCEPTS_A_TK')
    TURNOVER_DYNAMIC_IL_1M = request.form.get('TURNOVER_DYNAMIC_IL_1M')
    CNT_TRAN_AUT_TENDENCY1M = request.form.get('CNT_TRAN_AUT_TENDENCY1M')
    SUM_TRAN_AUT_TENDENCY1M = request.form.get('SUM_TRAN_AUT_TENDENCY1M')
    AMOUNT_RUB_SUP_PRC = request.form.get('AMOUNT_RUB_SUP_PRC')
    PRC_ACCEPTS_A_AMOBILE = request.form.get('PRC_ACCEPTS_A_AMOBILE')
    SUM_TRAN_AUT_TENDENCY3M = request.form.get('SUM_TRAN_AUT_TENDENCY3M')
    CLNT_TRUST_RELATION = request.form.get('CLNT_TRUST_RELATION')
    PRC_ACCEPTS_TK = request.form.get('PRC_ACCEPTS_TK')###################
    PRC_ACCEPTS_A_MTP = request.form.get('PRC_ACCEPTS_A_MTP')
    REST_DYNAMIC_FDEP_1M = request.form.get('REST_DYNAMIC_FDEP_1M')
    PRC_ACCEPTS_A_MTP = request.form.get('PRC_ACCEPTS_A_MTP')
    CNT_TRAN_AUT_TENDENCY3M =request.form.get('CNT_TRAN_AUT_TENDENCY3M')
    CNT_ACCEPTS_TK=request.form.get('CNT_ACCEPTS_TK')
    APP_MARITAL_STATUS=request.form.get('APP_MARITAL_STATUS')
    REST_DYNAMIC_SAVE_3M=request.form.get('REST_DYNAMIC_SAVE_3M')
    CR_PROD_CNT_VCU=request.form.get('CR_PROD_CNT_VCU')
    REST_AVG_CUR=request.form.get('REST_AVG_CUR')
    CNT_TRAN_MED_TENDENCY1M=request.form.get('CNT_TRAN_MED_TENDENCY1M')
    APP_KIND_OF_PROP_HABITATION=request.form.get('APP_KIND_OF_PROP_HABITATION')
    CLNT_JOB_POSITION_TYPE=request.form.get('CLNT_JOB_POSITION_TYPE')
    AMOUNT_RUB_NAS_PRC=request.form.get('AMOUNT_RUB_NAS_PRC')
    CLNT_JOB_POSITION=request.form.get('CLNT_JOB_POSITION')
    APP_DRIVING_LICENSE=request.form.get('APP_DRIVING_LICENSE')
    TRANS_COUNT_SUP_PRC=request.form.get('TRANS_COUNT_SUP_PRC')
 

    d_dict = {'CR_PROD_CNT_IL': [0], 'AMOUNT_RUB_CLO_PRC': [0], 'PRC_ACCEPTS_A_EMAIL_LINK': [0], 'APP_REGISTR_RGN_CODE': [0],
              'PRC_ACCEPTS_A_POS': [0], 'PRC_ACCEPTS_A_TK': [0], 'TURNOVER_DYNAMIC_IL_1M': [0], 'CNT_TRAN_AUT_TENDENCY1M': [0],
              'SUM_TRAN_AUT_TENDENCY1M': [0], 'AMOUNT_RUB_SUP_PRC': [0], 'SUM_TRAN_AUT_TENDENCY3M': [0], 'CLNT_TRUST_RELATION': [0],
              'PRC_ACCEPTS_TK': [0], 'PRC_ACCEPTS_A_MTP': [0], 'REST_DYNAMIC_FDEP_1M': [0],
              'CNT_TRAN_AUT_TENDENCY3M': [0], 'CNT_ACCEPTS_TK': [0], 'APP_MARITAL_STATUS': [0],
              'REST_DYNAMIC_SAVE_3M': [0], 'CR_PROD_CNT_VCU': [0], 'REST_AVG_CUR': [0],
              'CNT_TRAN_MED_TENDENCY1M': [0], 'APP_KIND_OF_PROP_HABITATION': [0], 'CLNT_JOB_POSITION_TYPE': [0],
              'CLNT_JOB_POSITION': [0], 'APP_DRIVING_LICENSE': [0], 'TRANS_COUNT_SUP_PRC': [0]}

    return pd.DataFrame.from_dict(d_dict, orient='columns')

def feature_imp(model, data):
    importances = model
    indices = np.argsort(importances)[::-1]
    top_30 = indices[:30]
    data = data.iloc[:, top_30]
    return data

def min_max_scale(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    #scaler.fit(data)
    data_scaled = scaler.fit_transform(data.values.reshape(30, -1))
    data = data_scaled.reshape(-1, 30)
    return pd.DataFrame(data)

@app.route('/send', methods=['POST'])
def show_data():
    df = get_data()
    featured_data = feature_imp(model, df)
    #input_df = pd.DataFrame([inputs], columns=feature_names)
    prediction = model.predict(df)
    churn_probability = model.predict_proba(input_df)[:, 1][0]
    if prediction[0] == 0:
        outcome ='No Churn'
    else:
        outcome= 'Churn'
    #scaled_data = min_max_scale(featured_data)
    #prediction = model.predict(scaled_data)
    #outcome = 'Churner'
    #if prediction == 0:
        #outcome = 'Non-Churner'

    return render_template('results.html', tables = [df.to_html(classes='data', header=True)],
                           result = outcome)



if __name__=="__main__":
    app.run(debug=True)