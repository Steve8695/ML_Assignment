import streamlit as st
import pandas as pd
import pickle
 
# Load pre-trained model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
 
# Define feature names
feature_names = ['CR_PROD_CNT_IL', 'AMOUNT_RUB_CLO_PRC', 'PRC_ACCEPTS_A_EMAIL_LINK',
       'APP_REGISTR_RGN_CODE', 'PRC_ACCEPTS_A_POS', 'PRC_ACCEPTS_A_TK',
       'TURNOVER_DYNAMIC_IL_1M', 'CNT_TRAN_AUT_TENDENCY1M',
       'SUM_TRAN_AUT_TENDENCY1M', 'AMOUNT_RUB_SUP_PRC',
       'PRC_ACCEPTS_A_AMOBILE', 'SUM_TRAN_AUT_TENDENCY3M',
       'CLNT_TRUST_RELATION', 'PRC_ACCEPTS_TK', 'PRC_ACCEPTS_A_MTP',
       'REST_DYNAMIC_FDEP_1M', 'CNT_TRAN_AUT_TENDENCY3M', 'CNT_ACCEPTS_TK',
       'APP_MARITAL_STATUS', 'REST_DYNAMIC_SAVE_3M', 'CR_PROD_CNT_VCU',
       'REST_AVG_CUR', 'CNT_TRAN_MED_TENDENCY1M',
       'APP_KIND_OF_PROP_HABITATION', 'CLNT_JOB_POSITION_TYPE',
       'AMOUNT_RUB_NAS_PRC', 'APP_DRIVING_LICENSE', 'TRANS_COUNT_SUP_PRC',
       'APP_EDUCATION', 'CNT_TRAN_CLO_TENDENCY1M', 'SUM_TRAN_MED_TENDENCY1M',
       'PRC_ACCEPTS_A_ATM', 'PRC_ACCEPTS_MTP', 'TRANS_COUNT_NAS_PRC',
       'APP_TRAVEL_PASS', 'CNT_ACCEPTS_MTP', 'CR_PROD_CNT_TOVR', 'APP_CAR',
       'CR_PROD_CNT_PIL', 'SUM_TRAN_CLO_TENDENCY1M', 'APP_POSITION_TYPE',
       'TURNOVER_CC', 'TRANS_COUNT_ATM_PRC', 'AMOUNT_RUB_ATM_PRC',
       'TURNOVER_PAYM', 'AGE', 'CNT_TRAN_MED_TENDENCY3M', 'CR_PROD_CNT_CC',
       'SUM_TRAN_MED_TENDENCY3M', 'REST_DYNAMIC_FDEP_3M', 'REST_DYNAMIC_IL_1M',
       'APP_EMP_TYPE', 'SUM_TRAN_CLO_TENDENCY3M', 'CR_PROD_CNT_CCFP',
       'CNT_TRAN_CLO_TENDENCY3M', 'REST_DYNAMIC_CUR_1M', 'REST_AVG_PAYM',
       'APP_COMP_TYPE', 'LDEAL_GRACE_DAYS_PCT_MED', 'REST_DYNAMIC_CUR_3M',
       'CNT_TRAN_SUP_TENDENCY3M', 'TURNOVER_DYNAMIC_CUR_1M',
       'REST_DYNAMIC_PAYM_3M', 'SUM_TRAN_SUP_TENDENCY3M', 'REST_DYNAMIC_IL_3M',
       'CNT_TRAN_ATM_TENDENCY3M', 'CNT_TRAN_ATM_TENDENCY1M',
       'TURNOVER_DYNAMIC_IL_3M', 'SUM_TRAN_ATM_TENDENCY3M',
       'DEAL_GRACE_DAYS_ACC_S1X1', 'DEAL_YWZ_IR_MIN',
       'SUM_TRAN_SUP_TENDENCY1M', 'DEAL_YWZ_IR_MAX', 'SUM_TRAN_ATM_TENDENCY1M',
       'REST_DYNAMIC_PAYM_1M', 'CNT_TRAN_SUP_TENDENCY1M',
       'DEAL_GRACE_DAYS_ACC_AVG', 'TURNOVER_DYNAMIC_CUR_3M', 'PACK',
       'CLNT_SETUP_TENOR', 'DEAL_GRACE_DAYS_ACC_MAX',
       'TURNOVER_DYNAMIC_PAYM_3M', 'TURNOVER_DYNAMIC_PAYM_1M',
       'TRANS_AMOUNT_TENDENCY3M', 'TRANS_CNT_TENDENCY3M', 'REST_DYNAMIC_CC_1M',
       'LDEAL_USED_AMT_AVG_YWZ', 'TURNOVER_DYNAMIC_CC_1M',
       'LDEAL_ACT_DAYS_ACC_PCT_AVG', 'REST_DYNAMIC_CC_3M', 'MED_DEBT_PRC_YWZ',
       'LDEAL_ACT_DAYS_PCT_TR3', 'LDEAL_ACT_DAYS_PCT_AAVG',
       'LDEAL_DELINQ_PER_MAXYWZ', 'TURNOVER_DYNAMIC_CC_3M',
       'LDEAL_ACT_DAYS_PCT_TR', 'LDEAL_ACT_DAYS_PCT_TR4',
       'LDEAL_ACT_DAYS_PCT_CURR', 'CLNT_JOB_POSITION']
 
# Create Streamlit app
st.title('Churn Prediction')
 
# Create input form
with st.form('input_form'):
    inputs = []
    for feature in feature_names:
        input_val = st.number_input(feature, min_value=0.0, value=0.0)
        inputs.append(input_val)
 
    # Create submit button
    submit_button = st.form_submit_button('Predict Churn')
 
# Make prediction on submit
if submit_button:
    input_df = pd.DataFrame([inputs], columns=feature_names)
    prediction = model.predict(input_df)
    churn_probability = model.predict_proba(input_df)[:, 1][0]
 
    # Display results
    st.write('**Prediction:**')
    if prediction[0] == 0:
        st.write('No Churn')
    else:
        st.write('Churn')