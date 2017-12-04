# -*- coding: utf-8 -*-
import sys
import os
import re
import time
import datetime
import random
import matplotlib.pyplot as plt
import seaborn

from functools import partial

import pandas as pd
import numpy as np

from multiprocessing import Pool
from keras.utils import np_utils

####### Converting data Format ##############
# 파일　Path 설정
OS_PATH = os.path.abspath('./')
FIND_PATH = re.compile('prostate_cancer')
BASE_PATH = OS_PATH[:FIND_PATH.search(OS_PATH).span()[1]]
sys.path.append(BASE_PATH)
# Directory Path
DATA_DIR = os.path.join(BASE_PATH, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')  # raw data 폴더
PREP_DIR = os.path.join(DATA_DIR, 'prep')
# Raw data Path
RAW_EXCEL_PATH = os.path.join(RAW_DATA_DIR, '전립선_연구자료.xlsx')  # 전립선　연구　총체
RAW_LAB_PATH = os.path.join(
    RAW_DATA_DIR, 'lab_test.txt')  # labtest result table
RAW_PRESCRIBE_PATH = os.path.join(
    RAW_DATA_DIR, 'prescribe.xlsx')  # prescribe file
KCD_PATH = os.path.join(RAW_DATA_DIR, 'KCD.xlsx')  # KCD code정리된것
MEDICINE_CONTEXT_PATH = os.path.join(RAW_DATA_DIR, 'medicine_context.xlsx')
LAB_LIST_PATH = os.path.join(RAW_DATA_DIR, 'lab코드 목록.xlsx')
# preprocessed data output path
PRESCRIBE_PATH = os.path.join(PREP_DIR, 'prescribe.h5')
LABTEST_PATH = os.path.join(PREP_DIR, 'labtest.h5')
DIAG_PATH = os.path.join(PREP_DIR, 'diagnosis.h5')
# The Constants in Data
START_DATE = '2002-01'
END_DATE = '2017-08'
DATE_RANGE = pd.date_range(START_DATE, END_DATE, freq='m').map(
    lambda x: x.year*100 + x.month)

LABEL_CODE_LIST = ['L3718', 'L3593', 'L359301']  # PSA 검사　관련　코드

# EMERGENCY CODE와　Non-Emergency code매칭
EMGCY_AND_NOT_DICT = {
    'L8123': 'L3123',  # 뇨검사（단백정량）
    'L8124': 'L3124',  # 뇨검사（소디움）
    'L8125': 'L3125',  # 뇨검사（염소）
    'L8126': 'L3126',  # 뇨검사（포타슘）
    'L8011': 'L2011',  # 백혈구수
    'L8012': 'L2012',  # 적혈구수
    'L8013': 'L2013',  # 혈색소（광전비색법）
    'L8014': 'L2014',  # 헤마토크리트
    'L801401': 'L201401',
    'L801402': 'L201402',
    'L801403': 'L201403',
    'L8015': 'L2015',  # 적혈구분포계수
    'L81051': 'L61021',  # 체액일반검사
    'L8016': 'L2017',  # 혈소판수
    'L80161': 'L80161',  # 평균혈소판용적
    'L80162': 'L20162',
    'L80163': 'L20163',  # 　혈소판용적비율
    'L8017': 'L2016',  # 혈소판분포계수
    'L8018': 'L2018',  # 백혈구백분율
    'L801806': 'L201806',
    'L801807': 'L201807',
    'L801808': 'L201808',
    'L801809': 'L201809',
    'L801810': 'L201810',
    'L8021': 'L2111',  # 프로트롬비시간
    'L8022': 'L8022',  # 부분트롬보프라스틴검사
    'L8031': 'L3011',  # 총단백
    'L8032': 'L3012',  # 알부민
    'L8034': 'L3015',  # 혈청GOT
    'L8035': 'L3016',  # 혈청GPT
    'L8036': 'L3018',  # 당정량
    'L8037': 'L3019',  # 요소질소
    'L8038': 'L3020',  # 크레아티닌
    'L8039': 'L3024',  # 식후당정량
    'L8040': 'L3025',  # 당정량， Random
    'L8041': 'L3041',  # 소디움
    'L8042': 'L3042',  # 포타슘
    'L8043': 'L3043',  # 염소
    'L8044': 'L3044',  # 혈액총탄산
    'L8046': 'L3022',  # 총칼슘
    'L8047': 'L3023',  # 인
    'L8048': 'L3021',  # 요산
    'L8049': 'L3013',  # 콜레스테롤
    'L8050': 'L3029',  # 중성지방
    'L8053': 'L3057',  # LDH
    'L8057': 'L3067',  # 리파아제
    'L8059': 'L3032',  # 마그네슘
}

MEDICINE_CONTEXT_COLS = ['medi_code', 'medi_name',
                         's_date', 'e_date', 'ingd', 'ATC_code', 'ATC_desc']
RAW_PRESCRIBE_COLS = ['no', 'medi_code', 'g_code', 'g_code_name',
                      'medi_name', 'start_date', 'total', 'once', 'times', 'day']
RAW_DIAGNOSIS_COLS = ['no', 'start_date',
                      'end_date', 'seg', 'KCD_code', 'description']
# time period structure
INPUT_PERIOD = 36
GAP_PERIOD = 3
PREDICTION_PERIOD = 9


def set_input_period(period):
    global INPUT_PERIOD
    INPUT_PERIOD = period


def set_gap_period(period):
    global GAP_PERIOD
    GAP_PERIOD = period


def set_prediction_period(period):
    global PREDICTION_PERIOD
    PREDICTION_PERIOD = period
# the number of cpu core (Multi-processing)
CORE_NUMS = 12


def set_core_nums(nums):
    global CORE_NUMS
    CORE_NUMS = nums

####### Pre-Processing Data ###############
'''
    Raw Data를 정규화하여, HDF5 format으로 저장
    HDF5 : 파일 저장 포맷, 계층적 구조로 빠르게 필요한 데이터만을 입출력할 수 있음
    
    메소드 구성
    - preprocess_labtest 
        : 환자의 검사 결과 dataset을 전처리하는 코드
        
        Related method
            - change_number
                : 숫자　형태가　아닌　결과값을　숫자형태로　바꾸어줌 

            - normalize_number
                : 결과값을　정규화해주는　코드
                - revise_avg
                - revise_std
                - revise_min
                - revise_max
            
            - set_mapping_table
                : 정규화　때　이용된　값（평균，표준편차，최대，최소）를　저장

        Issue
            1. 같은 검사이더라도, 코드가 나누어져 있는 경우가 있음
                그것을 통일시켜 주어야 함
            2. 검사의 분포 형태는 정규분포가 아님，
                정규분포 꼴로 바꾸어주어야 함
            3. 다양한 검사 코드가 있음. 이 중에서 순서를 잘 잡아주어야 함

    - preprocess_prescribe
        : 환자에게　내려진　약물　처방　코드를　저장

    - preprocess_diagnosis
        : 환자에게　내려진　진단명을　저장
        
    - preprocess_label
        : 예측해야할　값（전립선암）을　저장

'''


def preprocess_labtest():
    global RAW_LAB_PATH, LABTEST_PATH
    # 0. data Loading
    lab_df = pd.read_csv(RAW_LAB_PATH, delimiter='\t', header=None, names=[
                         'no', 'labtest', 'date', 'result'])
    # 1. datetime format : 20170511 -> 201705
    lab_df.loc[:, "date"] = pd.to_datetime(
        lab_df.loc[:, "date"].astype(str), format='%Y%m%d')
    lab_df.loc[:, 'date'] = lab_df.date.map(lambda x: x.year*100+x.month)
    # 2. replace emergency code
    lab_df.loc[:, 'labtest'] = lab_df.loc[:, 'labtest'].map(
        lambda x: EMGCY_AND_NOT_DICT[x] if x in EMGCY_AND_NOT_DICT else x)
    # 3. convert result from string to float
    lab_df.loc[:, 'result'] = lab_df.loc[
        :, 'result'].map(change_number).dropna()
    # 4. drop null value
    lab_df = lab_df.dropna()
    # 5. 각 코드별로 min~max가 다르므로, 각 코드별 정규화을 위한 mapping table 생성
    mapping_table = set_mapping_table(lab_df)
    # 6. 각 검사별로 normalizing 적용
    lab_list = []
    for lab_name in lab_df.labtest.unique():
        _lab_df = lab_df.loc[lab_df.labtest == lab_name]
        r_avg = mapping_table['AVG'][lab_name]
        r_min = mapping_table['MIN'][lab_name]
        r_max = mapping_table['MAX'][lab_name]
        _lab_df.loc[:, 'result'] = _lab_df.result.map(
            normalize_number(r_avg, r_min, r_max))
        lab_list.append(_lab_df)
    normalized_lab_df = pd.concat(lab_list)
    normalized_lab_df.dropna(inplace=True)
    normalized_lab_df.to_hdf(LABTEST_PATH, 'prep',
                             format='table', data_columns=True, mode='a')

    # 7. count_lab_df : lab_test 별로 몇건이 있는지 저장
    count_lab_df = normalized_lab_df[
        ['no', 'labtest']].groupby('labtest').count()
    count_lab_df.columns = ['counts']
    count_lab_df = count_lab_df.sort_values('counts', ascending=False)
    count_lab_df.to_hdf(LABTEST_PATH, 'metadata/usecol',
                        format='table', data_columns=True, mode='a')


def preprocess_label():
    global DIAG_PATH, RAW_EXCEL_PATH, RAW_DIAGNOSIS_COLS
    # 진단　정보
    diag_df = pd.read_excel(RAW_EXCEL_PATH, sheetname=2)
    diag_df.columns = RAW_DIAGNOSIS_COLS
    prostate_df = diag_df[diag_df.KCD_code == 'C61']
    prostate_df.sort_values(['no', 'start_date'], inplace=True)
    prostate_df = prostate_df.drop_duplicates(['no', 'KCD_code'], keep='first')
    prostate_df.loc[:, 'start_date'] = prostate_df.loc[:, 'start_date'] // 100
    prostate_df.to_hdf(DIAG_PATH,
                       'prep', format='table', data_columns=True, mode='a')


def preprocess_prescribe():
    global RAW_PRESCRIBE_PATH, RAW_PRESCRIBE_COLS
    raw_pres_df = pd.read_excel(RAW_PRESCRIBE_PATH)
    raw_pres_df.columns = RAW_PRESCRIBE_COLS

    raw_pres_df.loc[:, 'start_date'] = pd.to_datetime(
        raw_pres_df.start_date.astype(str), format='%Y%m%d')
    raw_pres_df.day = raw_pres_df.day.astype(int)
    raw_pres_df.loc[:, 'end_date'] = raw_pres_df.start_date + \
        raw_pres_df.day.map(lambda x: np.timedelta64(x, 'D'))

    raw_pres_df.loc[:, 'start_date'] = raw_pres_df.start_date.map(
        lambda x: x.year*100+x.month)  # EX) 20170211 -> 201702
    raw_pres_df.loc[:, 'end_date'] = raw_pres_df.end_date.map(
        lambda x: x.year*100+x.month)  # EX) 20170211 -> 201702

    pres_df = raw_pres_df[['no', 'g_code', 'start_date', 'end_date']]
    pres_df = pres_df.drop_duplicates()
    pres_df.to_hdf(PRESCRIBE_PATH, 'prep', format='table',
                   data_columns=True, mode='a')
    pd.DataFrame(raw_pres_df.g_code.value_counts()).to_hdf(
        PRESCRIBE_PATH, 'metadata/mapping_table', format='table', data_columns=True, mode='a')


def preprocess_diagnosis():
    global RAW_DIAGNOSIS_PATH, DIAG_PATH
    raw_diag_df = pd.read_excel(RAW_DIAGNOSIS_PATH)
    raw_diag_df.columns = ['no', 'date',
                           'end_date', 'seg', 'KCD_code', 'en', 'kr']
    raw_diag_df.loc[:, 'date'] = pd.to_datetime(
        raw_diag_df.date.astype(str), format='%Y%m%d')
    raw_diag_df.loc[:, 'date'] = raw_diag.df.date.map(
        lambda x: x.year*100 + x.month)  # EX) 20170211 -> 201702

    diag_df = raw_diag_df[['no', 'date', 'KCD_code']].drop_duplicates()
    diag_df.to_hdf(DIAG_PATH, 'data', format='table',
                   data_columns=True, mode='a')
    pd.DataFrame(diag_df.KCD_code.value_counts()).to_hdf(
        DIAG_PATH, 'metadata/mapping_table', format='table', data_columns=True, mode='a')


def change_number(x):
    # 숫자 표현을 통일  (범위 쉼표 등 표현을 단일표현으로 통일）
    str_x = str(x).replace(" ", "")
    str_x = str_x.replace("<", "")
    str_x = str_x.replace(">", "")
    # 숫자로 구성된 데이터를 float로 바꾸어 줌
    re_num = re.compile('^[+-]{0,1}[\d\s]+[.]{0,1}[\d\s]*$')
    # 쉼표(,)가 있는 숫자를 선별
    re_comma = re.compile('^[\d\s]*,[\d\s]*[.]{0,1}[\d\s]*$')
    # 범위(~,-)가 있는 숫자를 선별
    re_range = re.compile('^[\d\s]*[~\-][\d\s]*$')
    if re_num.match(str_x):
        return float(str_x)
    else:
        if re_comma.match(str_x):
            return change_number(str_x.replace(',', ""))
        elif re_range.match(str_x):
            if "~" in str_x:
                a, b = str_x.split("~")
            else:
                a, b = str_x.split("-")
            return np.mean((change_number(a), change_number(b)))
        else:
            return np.nan


def normalize_number(mean_x, min_x, max_x):
    '''
    dataframe 내 이상값을 전처리하는 함수.
    dataframe.map 을 이용할 것이므로, 함수 in 함수 구조 사용
    '''
    def _normalize_number(x):
        str_x = str(x).strip()

        re_num = re.compile('^[+-]?[\d]+[.]?[\d]*$')
        re_lower = re.compile('^<[\d\s]*[.]{0,1}[\d\s]*$')
        re_upper = re.compile('^>[\d\s]*[.]{0,1}[\d\s]*$')
        re_star = re.compile('^[\s]*[*][\s]*$')
        if re_num.match(str_x):
            # 숫자형태일경우
            float_x = np.float(str_x)
            if float_x > max_x:
                return 1
            elif float_x < min_x:
                return 0
            else:
                return (np.float(str_x) - min_x)/(max_x-min_x)
        else:
            if re_lower.match(str_x):
                return 0
            elif re_upper.match(str_x):
                return np.float(1)
            elif re_star.match(str_x):
                return np.float((mean_x-min_x)/(max_x-min_x))
            else:
                return np.nan
    return _normalize_number


def revise_avg(x):
    # 10~90% 내에 있는 값을 이용해서 평균 계산
    quan_min = x.quantile(0.10)
    quan_max = x.quantile(0.90)
    return x[(x > quan_min) & (x < quan_max)].mean()


def revise_std(x):
    # 1~99% 내에 있는 값을 이용해서 표준편차 계산
    quan_min = x.quantile(0.01)
    quan_max = x.quantile(0.99)
    return x[(x > quan_min) & (x < quan_max)].std()


def revise_min(x):
    # 3시그마 바깥 값과 quanter 값의 사이값으로 결정
    # 3 시그마 바깥 값
    std_min = revise_avg(x)-revise_std(x) * 3
    q_min = x.quantile(0.01)
    if std_min < 0:
        # 측정값중에서 음수가 없기 때문에, 음수인 경우는 고려안함
        return q_min
    else:
        return np.mean((std_min, q_min))


def revise_max(x):
    # 3시그마 바깥 값과 quanter 값의 사이값으로 결정
    std_max = revise_avg(x)+revise_std(x)*3
    q_max = x.quantile(0.99)
    return np.mean((std_max, q_max))


def set_mapping_table(lab_df):
    '''
    labtest의 mapping table을 생성하는 함수
    평균/ 최솟값/최댓값으로 구성
    이를 hdf5파일의 metadata에 저장
    '''
    global LABTEST_PATH

    result_df = pd.DataFrame(columns=['lab_test', 'AVG', 'MIN', 'MAX'])
    for lab_name in lab_df.labtest.unique():
        per_lab_df = lab_df.loc[lab_df.labtest == lab_name, ['no', 'result']]
        # 2. 이상 값 처리 시 대응되는 값
        r_avg = revise_avg(per_lab_df.result)
        r_min = revise_min(per_lab_df.result)
        r_max = revise_max(per_lab_df.result)
        # 3. save
        result_df = result_df.append(
            {'lab_test': lab_name, 'AVG': r_avg, 'MIN': r_min, 'MAX': r_max}, ignore_index=True)

    result_df = result_df.set_index('lab_test')
    result_df = result_df.sort_index()
    result_df.to_hdf(LABTEST_PATH, 'metadata/mapping_table',
                     format='table', data_columns=True, mode='a')
    return result_df.to_dict()

####### Converting Data From long Format to Time-Serial Format ###############
'''
    Raw Data를 정규화하여, HDF5 format으로 저장
    HDF5 : 파일 저장 포맷, 계층적 구조로 빠르게 필요한 데이터만을 입출력할 수 있음
    
    메소드 구성
    - get_timeserial_lab_df 
        : 환자의 검사 결과를　반환
        Arguments
            no : 해당 환자번호
            labtest_list : 가져 올 시험결과 리스트

    - get_timeserial_pres_df
        : 환자에게 내려진 약물 처방 코드를 반환
        Arguments
            no : 해당 환자번호

    - get_timeserial_diag_df
        : 환자에게 내려진 진단명을 반환
        Arguments
            no : 해당 환자번호
    
    - get_timeserial_label_df
        : 예측해야할 값(전립선암)을 반환
        Argument
            no : 해당　환자번호
            mode : 반환의　형태
'''
def get_timeserial_lab_df(no, labtest_list):
    global LABTEST_PATH, DATE_RANGE
    lab_store = pd.HDFStore(LABTEST_PATH, mode='r')
    try:
        lab_df = lab_store.select('prep', where='no=={}'.format(no))
    finally:
        lab_store.close()

    # 같은　월　평균　값
    mean_y = lab_df.loc[lab_df.labtest.isin(labtest_list),
                        ['date', 'labtest', 'result']]\
        .groupby(['date', 'labtest'])\
        .mean()\
        .reset_index()\
        .pivot_table(index=['labtest'], columns=['date'])
    mean_y.columns = mean_y.columns.droplevel()
    mean_y = mean_y.reindex(index=labtest_list, columns=DATE_RANGE)
    # 같은　월　최소　값
    min_y = lab_df.loc[lab_df.labtest.isin(labtest_list),
                       ['date', 'labtest', 'result']]\
        .groupby(['date', 'labtest'])\
        .min()\
        .reset_index()\
        .pivot_table(index=['labtest'], columns=['date'])
    min_y.columns = min_y.columns.droplevel()
    min_y = min_y.reindex(index=labtest_list, columns=DATE_RANGE)
    # 같은　월　최대　값
    max_y = lab_df.loc[lab_df.labtest.isin(labtest_list),
                       ['date', 'labtest', 'result']]\
        .groupby(['date', 'labtest'])\
        .max()\
        .reset_index()\
        .pivot_table(index=['labtest'], columns=['date'])
    max_y.columns = max_y.columns.droplevel()
    max_y = max_y.reindex(index=labtest_list, columns=DATE_RANGE)
    # 같은　월　결측　유무
    bool_y = mean_y.isnull().astype(np.int)

    #total_y = np.stack([mean_y,min_y,max_y,bool_y])
    return mean_y, min_y, max_y, bool_y


def get_timeserial_pres_df(no):
    global PRESCRIBE_PATH, DATE_RANGE

    pres_store = pd.HDFStore(PRESCRIBE_PATH, mode='r')
    try:
        pres_df = pres_store.select('prep', where='no=={}'.format(no))
        usecol = pres_store.select('metadata/mapping_table').index
    finally:
        pres_store.close()

    ts_pres_df = pres_df.pivot_table(index=['g_code'], columns=[
                                     'start_date'], values=['no'])
    ts_pres_df.columns = ts_pres_df.columns.droplevel()
    ts_pres_df = ts_pres_df.reindex(columns=DATE_RANGE, index=usecol)

    for _, row in pres_df.sort_values(['no', 'g_code', 'start_date']).iterrows():
        ts_pres_df.loc[row.g_code].loc[row.start_date:row.end_date] = 1

    ts_pres_df = ts_pres_df.fillna(0)
    ts_pres_df[ts_pres_df != 0] = 1

    assert pd.isnull(ts_pres_df).sum().sum(
    ) == 0, "the return value of get_timeserial_pres_df contains null value"

    return ts_pres_df


def get_timeserial_diag_df(no):
    global DIAG_PATH, DATE_RANGE
    diag_store = pd.HDFStore(DIAG_PATH)
    try:
        diag_df = diag_store.select('data', where='no=={}'.format(no))
        usecol = diag_store.select('metadata/mapping_table').index
    finally:
        diag_store.close()

    ts_diag_df = diag_df.pivot_table(index=['KCD_code'], columns=['date'])
    ts_diag_df[~ts_diag_df.isnull()] = 1

    ts_diag_df.columns = ts_diag_df.columns.droplevel()
    ts_diag_df = ts_diag_df.reindex(columns=DATE_RANGE, index=usecol)
    ts_diag_df = ts_diag_df.fillna(0)

    return ts_diag_df


def get_timeserial_label_df(no, mode='list'):
    global DIAG_PATH, DATE_RANGE
    label_store = pd.HDFStore(DIAG_PATH, mode='r')
    try:
        prostate_df = label_store.select('prep', where='no=={}'.format(no))
    finally:
        label_store.close()
    label_df = prostate_df.loc[:, ['no', 'start_date', 'KCD_code']]\
        .pivot_table(index=['KCD_code'], columns=['start_date'])\
        .reset_index()

    label_df.columns = label_df.columns.droplevel()

    if mode == 'list':
        return label_df
    elif mode == 'series':
        label_ts_df = label_df.reindex(columns=DATE_RANGE)\
            .fillna(method='ffill', axis=1)\
            .fillna(0)
        return label_ts_df
    else:
        return label_df
####### Constructing Input Dataset ###############
'''
    모델이 학습할 수 있도록 그 형태에 맞춰 데이터셋을 구성하는 메소드 집합
    연산량이 많기 때문에 효율을 위해 multi-processing 이용
    - get_dataset
        : 데이터셋을 구성하는 함수
        Arguments
        lab_nums : PSA 검사를 제외하고, 추가적으로input 데이터로 사용할 lab test의 갯수(0 : PSA 검사 3개만 들어감)
        psa_cr : 최소 psa검사수
        train_df : 라벨이 된 환자 dataframe

        내부 메소드
        비 환자가 환자에 비해 데이터 양이 더 많음．
        비환자의 데이터 중 중복이 되는 것들을 제외함으로서, 데이터 양을 줄임
        _get_disease_dataset :
            질병이 걸린 환자를 대상으로 데이터셋을 구성하는 함수
        _get_nondisease_dataset :
            질병이 걸리지 않은 환자를 대상으로 데이터셋을 구성하는 함수

        Labtest 데이터에　한에서，데이터 셋의　구성을 Min,avg,Max,Bool로　구성
        이는 한달 내에 중복으로 검사받은 경우, 그 결과 중 대표값들을 최대한 살리고자 함임
'''
def get_dataset(lab_nums, psa_cr, train_df):
    global LABTEST_PATH, DIAG_PATH, CORE_NUMS, LABEL_CODE_LIST
    patient_psa_count = 0
    patient_all_count = 0
    # 환자의 진단일 불러오기
    label_store = pd.HDFStore(DIAG_PATH, mode='r')
    try:
        prostate_df = label_store.select('prep', columns=['no', 'start_date'])
    finally:
        label_store.close()

    train_y_df = train_df[train_df.pc_result == 'Y']
    train_n_df = train_df[train_df.pc_result == 'N']

    lab_store = pd.HDFStore(LABTEST_PATH, mode='r')
    try:
        usecol = lab_store.select('metadata/usecol')
        mapping_table = lab_store.select('metadata/mapping_table')
    finally:
        lab_store.close()
    mapping_table = mapping_table.dropna()
    usecol = usecol.reindex(mapping_table.index).sort_values(
        'counts', ascending=False)
    # PSA　검사　３개와　Top lab_nums개를　붙임
    if lab_nums > 0:
        labname_list = np.append(
            LABEL_CODE_LIST, usecol.index[:lab_nums].values)
    else:
        labname_list = LABEL_CODE_LIST
    labname_count = len(labname_list)

    pool = Pool()
    start_time = time.time()
    result = pool.map_async(partial(
        _get_disease_dataset, labname_list, psa_cr), np.array_split(train_y_df, CORE_NUMS))
    y_lab = np.concatenate([lab for lab, diag, pres, label in result.get()])
    y_diag = np.concatenate([diag for lab, diag, pres, label in result.get()])
    y_pres = np.concatenate([pres for lab, diag, pres, label in result.get()])
    y_label = np.concatenate([label for lab, diag, pres, label in result.get()])
    print("{}--consumed".format(time.time() - start_time))
    pool.close()

    pool = Pool()
    start_time = time.time()
    result = pool.map_async(partial(
        _get_nondisease_dataset, labname_list, psa_cr), np.array_split(train_n_df, CORE_NUMS))
    n_lab = np.concatenate([lab for lab, diag, pres, label in result.get()])
    n_diag = np.concatenate([diag for lab, diag, pres, label in result.get()])
    n_pres = np.concatenate([pres for lab, diag, pres, label in result.get()])
    n_label = np.concatenate([label for lab, diag, pres, label in result.get()])
    print("{}--consumed".format(time.time() - start_time))
    pool.close()

    y_diag = np.expand_dims(y_diag, axis=-1)
    n_diag = np.expand_dims(n_diag, axis=-1)
    y_pres = np.expand_dims(y_pres, axis=-1)
    n_pres = np.expand_dims(n_pres, axis=-1)

    y_label = np_utils.to_categorical(y_label, 2)
    n_label = np_utils.to_categorical(n_label, 2)

    return y_lab, y_diag, y_pres, y_label, n_lab, n_diag, n_pres, n_label


def _get_disease_dataset(labname_list, psa_cr, train_df):
    global DATE_RANGE, INPUT_PERIOD, GAP_PERIOD, PREDICTION_PERIOD, DIAG_PATH, LABTEST_PATH
    # 환자의　진단일　불러오기
    label_store = pd.HDFStore(DIAG_PATH, mode='r')
    try:
        prostate_df = label_store.select('prep', columns=['no', 'start_date'])
    finally:
        label_store.close()

    lab_store = pd.HDFStore(LABTEST_PATH, mode='r')
    try:
        mapping_table = lab_store.select('metadata/mapping_table')
    finally:
        lab_store.close()
    mapping_table = mapping_table.dropna()
    # lab name 세기
    labname_count = len(labname_list)

    patient_psa_count = 0
    patient_all_count = 0
    lab_list = []
    diag_list = []
    pres_list = []
    for _, row in train_df.iterrows():
        no = row.no
        diag_date = prostate_df[prostate_df.no == no].iloc[0].start_date
        ts_lab_dfs = get_timeserial_lab_df(no, labname_list)
        ts_diag_df = get_timeserial_diag_df(no)
        ts_pres_df = get_timeserial_pres_df(no)
        diag_idx = DATE_RANGE.get_loc(diag_date)
        input_max_idx = diag_idx - (INPUT_PERIOD + GAP_PERIOD)
        if input_max_idx >= 0:
            input_min_idx = diag_idx - \
                (INPUT_PERIOD + GAP_PERIOD + PREDICTION_PERIOD)
            if input_min_idx < 0:
                input_min_idx = 0
            psa_no_check = False
            all_no_check = False
            for input_idx in range(input_min_idx, input_max_idx + 1):
                input_dfs = []
                # avg, min, max 값　imputation
                for ts_lab_df in ts_lab_dfs[:-1]:
                    input_df = ts_lab_df.loc[:, DATE_RANGE[
                        input_idx]:DATE_RANGE[input_idx+INPUT_PERIOD-1]]
                    psa_count = input_df.iloc[:3].count().sum()
                    all_count = input_df.count().sum()
                    input_df = input_df\
                        .fillna(method='bfill', axis=1)\
                        .fillna(method='ffill', axis=1)
                    for idx in input_df.index:
                        avg = (mapping_table.loc[idx].AVG - mapping_table.loc[idx].MIN) / (
                            mapping_table.loc[idx].MAX - mapping_table.loc[idx].MIN)
                        input_df.loc[idx, :] = input_df.loc[idx, :].fillna(avg)
                    input_dfs.append(input_df)
                    assert input_df.shape == (labname_count, INPUT_PERIOD)
                    if psa_count > psa_cr:
                        psa_no_check = True
                    if all_count > 10:
                        all_no_check = True
                # Boolean matrix
                input_df = ts_lab_dfs[-1].loc[:, DATE_RANGE[input_idx]
                    :DATE_RANGE[input_idx+INPUT_PERIOD-1]]
                assert input_df.shape == (labname_count, INPUT_PERIOD)
                input_dfs.append(input_df)

                if psa_no_check:
                    stack_np = []
                    for input_df in input_dfs:
                        # 결측이　있는지　검사
                        assert np.isnan(input_df.as_matrix()).sum() == 0, print(
                            "no : {} diag_idx : {}".format(no, input_idx))
                        input_np = input_df.as_matrix()
                        stack_np.append(input_np)
                    input_nps = np.stack(stack_np, axis=-1)
                    assert input_nps.shape == (labname_count, INPUT_PERIOD, 4)
                    lab_list.append(input_nps)
                    diag_np = ts_diag_df.loc[:, DATE_RANGE[input_idx]:DATE_RANGE[
                        input_idx+INPUT_PERIOD-1]].as_matrix()
                    diag_list.append(diag_np)
                    pres_np = ts_pres_df.loc[:, DATE_RANGE[input_idx]:DATE_RANGE[
                        input_idx+INPUT_PERIOD-1]].as_matrix()
                    pres_list.append(pres_np)
            if psa_no_check:
                patient_psa_count = patient_psa_count + 1
            if all_no_check:
                patient_all_count = patient_all_count + 1

    train_lab = np.stack(lab_list)
    train_diag = np.stack(diag_list)
    train_pres = np.stack(pres_list)

    labels = np.ones(train_lab.shape[0])

    return train_lab, train_diag, train_pres, labels


def _get_nondisease_dataset(labname_list, psa_cr, train_df):
    global DATE_RANGE, INPUT_PERIOD, GAP_PERIOD, PREDICTION_PERIOD, DIAG_PATH, LABTEST_PATH
    # 환자의　진단일　불러오기
    label_store = pd.HDFStore(DIAG_PATH, mode='r')
    try:
        prostate_df = label_store.select('prep', columns=['no', 'start_date'])
    finally:
        label_store.close()

    lab_store = pd.HDFStore(LABTEST_PATH, mode='r')
    try:
        mapping_table = lab_store.select('metadata/mapping_table')
    finally:
        lab_store.close()
    mapping_table = mapping_table.dropna()
    # lab name 세기
    labname_count = len(labname_list)

    patient_psa_count = 0
    patient_all_count = 0
    jump = 0
    lab_list = []
    diag_list = []
    pres_list = []
    for _, row in train_df.iterrows():
        no = row.no
        ts_lab_dfs = get_timeserial_lab_df(no, labname_list)
        ts_diag_df = get_timeserial_diag_df(no)
        ts_pres_df = get_timeserial_pres_df(no)
        psa_no_check = False
        all_no_check = False
        for input_idx in range(len(DATE_RANGE)-48):
            if jump > 0:
                jump -= 1
                continue
            input_dfs = []
            # avg, min, max 값　imputation
            for ts_lab_df in ts_lab_dfs[:-1]:
                input_df = ts_lab_df.loc[:, DATE_RANGE[
                    input_idx]:DATE_RANGE[input_idx+INPUT_PERIOD-1]]
                psa_count = input_df.iloc[:3].count().sum()
                all_count = input_df.count().sum()

                input_df = input_df\
                    .fillna(method='bfill', axis=1)\
                    .fillna(method='ffill', axis=1)
                for idx in input_df.index:
                    avg = (mapping_table.loc[idx].AVG - mapping_table.loc[idx].MIN) / (
                        mapping_table.loc[idx].MAX - mapping_table.loc[idx].MIN)
                    input_df.loc[idx, :] = input_df.loc[idx, :].fillna(avg)
                input_dfs.append(input_df)
                assert input_df.shape == (labname_count, INPUT_PERIOD)
                if psa_count > psa_cr:
                    psa_no_check = True
                if all_count > 10:
                    all_no_check = True
            # Boolean matrix
            input_df = ts_lab_dfs[-1].loc[:, DATE_RANGE[input_idx]
                :DATE_RANGE[input_idx+INPUT_PERIOD-1]]
            assert input_df.shape == (labname_count, INPUT_PERIOD)
            input_dfs.append(input_df)
            if psa_no_check:
                stack_np = []
                for input_df in input_dfs:
                    # 결측이　있는지　검사
                    assert np.isnan(input_df.as_matrix()).sum() == 0
                    input_np = input_df.as_matrix()
                    stack_np.append(input_np)
                input_nps = np.stack(stack_np, axis=-1)
                assert input_nps.shape == (labname_count, INPUT_PERIOD, 4)
                lab_list.append(input_nps)
                diag_np = ts_diag_df.loc[:, DATE_RANGE[input_idx]:DATE_RANGE[
                    input_idx+INPUT_PERIOD-1]].as_matrix()
                diag_list.append(diag_np)
                pres_np = ts_pres_df.loc[:, DATE_RANGE[input_idx]:DATE_RANGE[
                    input_idx+INPUT_PERIOD-1]].as_matrix()
                pres_list.append(pres_np)
                jump = 12
        if psa_no_check:
            patient_psa_count = patient_psa_count + 1
        if all_no_check:
            patient_all_count = patient_all_count + 1
    train_lab = np.stack(lab_list)
    train_diag = np.stack(diag_list)
    train_pres = np.stack(pres_list)

    labels = np.zeros(train_lab.shape[0])

    return train_lab, train_diag, train_pres, labels
