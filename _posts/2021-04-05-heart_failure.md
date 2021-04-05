---
layout: single
title: 'Python ML 연습문제 (심부전증 예방하기)'
---
# 데이터 분석으로 심부전증을 예방하기!

----------
## 데이터 소개
    - 다음 1개의 csv 파일을 사용합니다.
    heart_failure_clinical_records_dataset.csv
    
    - 각 파일의 컬럼은 아래와 같습니다.
    age: 환자의 나이
    anaemia: 환자의 빈혈증 여부 (0: 정상, 1: 빈혈)
    creatinine_phosphokinase: 크레아틴키나제 검사 결과
    diabetes: 당뇨병 여부 (0: 정상, 1: 당뇨)
    ejection_fraction: 박출계수 (%)
    high_blood_pressure: 고혈압 여부 (0: 정상, 1: 고혈압)
    platelets: 혈소판 수 (kiloplatelets/mL)
    serum_creatinine: 혈중 크레아틴 레벨 (mg/dL)
    serum_sodium: 혈중 나트륨 레벨 (mEq/L)
    sex: 성별 (0: 여성, 1: 남성)
    smoking: 흡연 여부 (0: 비흡연, 1: 흡연)
    time: 관찰 기간 (일)
    DEATH_EVENT: 사망 여부 (0: 생존, 1: 사망)


- 데이터 출처: https://www.kaggle.com/andrewmvd/heart-failure-clinical-data


## 도출결과

    - age: 환자의 나이
    - ejection_fraction: 박출계수 (%)
    - serum_creatinine: 혈중 크레아틴 레벨 (mg/dL)
    - smoking: 흡연 여부 (0: 비흡연, 1: 흡연)
    - 위의 4가지 요소들을 관리해주면 심부전증을 조금이나마 예방할 수 있음



## Step 1. 데이터셋 준비하기


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### Pandas 라이브러리로 csv파일 읽어들이기



```python
# pd.read_csv()로 csv파일 읽어들이기
df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>anaemia</th>
      <th>creatinine_phosphokinase</th>
      <th>diabetes</th>
      <th>ejection_fraction</th>
      <th>high_blood_pressure</th>
      <th>platelets</th>
      <th>serum_creatinine</th>
      <th>serum_sodium</th>
      <th>sex</th>
      <th>smoking</th>
      <th>time</th>
      <th>DEATH_EVENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75.0</td>
      <td>0</td>
      <td>582</td>
      <td>0</td>
      <td>20</td>
      <td>1</td>
      <td>265000.00</td>
      <td>1.9</td>
      <td>130</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55.0</td>
      <td>0</td>
      <td>7861</td>
      <td>0</td>
      <td>38</td>
      <td>0</td>
      <td>263358.03</td>
      <td>1.1</td>
      <td>136</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>65.0</td>
      <td>0</td>
      <td>146</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>162000.00</td>
      <td>1.3</td>
      <td>129</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.0</td>
      <td>1</td>
      <td>111</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>210000.00</td>
      <td>1.9</td>
      <td>137</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>65.0</td>
      <td>1</td>
      <td>160</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>327000.00</td>
      <td>2.7</td>
      <td>116</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

</div>



## Step 2. EDA 및 데이터 기초 통계 분석



```python
# 컬럼 분석하기 head(), info(), describe()
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 299 entries, 0 to 298
    Data columns (total 13 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   age                       299 non-null    float64
     1   anaemia                   299 non-null    int64  
     2   creatinine_phosphokinase  299 non-null    int64  
     3   diabetes                  299 non-null    int64  
     4   ejection_fraction         299 non-null    int64  
     5   high_blood_pressure       299 non-null    int64  
     6   platelets                 299 non-null    float64
     7   serum_creatinine          299 non-null    float64
     8   serum_sodium              299 non-null    int64  
     9   sex                       299 non-null    int64  
     10  smoking                   299 non-null    int64  
     11  time                      299 non-null    int64  
     12  DEATH_EVENT               299 non-null    int64  
    dtypes: float64(3), int64(10)
    memory usage: 30.5 KB



```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>anaemia</th>
      <th>creatinine_phosphokinase</th>
      <th>diabetes</th>
      <th>ejection_fraction</th>
      <th>high_blood_pressure</th>
      <th>platelets</th>
      <th>serum_creatinine</th>
      <th>serum_sodium</th>
      <th>sex</th>
      <th>smoking</th>
      <th>time</th>
      <th>DEATH_EVENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>299.000000</td>
      <td>299.000000</td>
      <td>299.000000</td>
      <td>299.000000</td>
      <td>299.000000</td>
      <td>299.000000</td>
      <td>299.000000</td>
      <td>299.00000</td>
      <td>299.000000</td>
      <td>299.000000</td>
      <td>299.00000</td>
      <td>299.000000</td>
      <td>299.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>60.833893</td>
      <td>0.431438</td>
      <td>581.839465</td>
      <td>0.418060</td>
      <td>38.083612</td>
      <td>0.351171</td>
      <td>263358.029264</td>
      <td>1.39388</td>
      <td>136.625418</td>
      <td>0.648829</td>
      <td>0.32107</td>
      <td>130.260870</td>
      <td>0.32107</td>
    </tr>
    <tr>
      <th>std</th>
      <td>11.894809</td>
      <td>0.496107</td>
      <td>970.287881</td>
      <td>0.494067</td>
      <td>11.834841</td>
      <td>0.478136</td>
      <td>97804.236869</td>
      <td>1.03451</td>
      <td>4.412477</td>
      <td>0.478136</td>
      <td>0.46767</td>
      <td>77.614208</td>
      <td>0.46767</td>
    </tr>
    <tr>
      <th>min</th>
      <td>40.000000</td>
      <td>0.000000</td>
      <td>23.000000</td>
      <td>0.000000</td>
      <td>14.000000</td>
      <td>0.000000</td>
      <td>25100.000000</td>
      <td>0.50000</td>
      <td>113.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>4.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>51.000000</td>
      <td>0.000000</td>
      <td>116.500000</td>
      <td>0.000000</td>
      <td>30.000000</td>
      <td>0.000000</td>
      <td>212500.000000</td>
      <td>0.90000</td>
      <td>134.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>73.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>60.000000</td>
      <td>0.000000</td>
      <td>250.000000</td>
      <td>0.000000</td>
      <td>38.000000</td>
      <td>0.000000</td>
      <td>262000.000000</td>
      <td>1.10000</td>
      <td>137.000000</td>
      <td>1.000000</td>
      <td>0.00000</td>
      <td>115.000000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>70.000000</td>
      <td>1.000000</td>
      <td>582.000000</td>
      <td>1.000000</td>
      <td>45.000000</td>
      <td>1.000000</td>
      <td>303500.000000</td>
      <td>1.40000</td>
      <td>140.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>203.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>95.000000</td>
      <td>1.000000</td>
      <td>7861.000000</td>
      <td>1.000000</td>
      <td>80.000000</td>
      <td>1.000000</td>
      <td>850000.000000</td>
      <td>9.40000</td>
      <td>148.000000</td>
      <td>1.000000</td>
      <td>1.00000</td>
      <td>285.000000</td>
      <td>1.00000</td>
    </tr>
  </tbody>
</table>

</div>




```python
df.columns
```




    Index(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
           'ejection_fraction', 'high_blood_pressure', 'platelets',
           'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
           'DEATH_EVENT'],
          dtype='object')



###  수치형 데이터의 히스토그램 그리기



```python
# seaborn의 histplot, jointplot, pairplot을 이용해 히스토그램 그리기
# 나이대 별 사망자의 수
sns.histplot(df,x='age',hue='DEATH_EVENT',kde='True' )
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f93876c6b38>



![image-center](/assets/images/output_10_1.png){: .align-center}




```python
sns.histplot(data=df.loc[df['creatinine_phosphokinase'] < 3000,'creatinine_phosphokinase'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9376b88630>




![image-center](/assets/images/output_11_1.png){: .align-center}




```python
# ejection_fraction 는 사망과 연관이 없다는걸 추측할 수 있음 
sns.histplot(x='ejection_fraction',data=df ,bins=13,hue='DEATH_EVENT',kde='True' )
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f937698cba8>





![image-center](/assets/images/output_12_1.png){: .align-center}




```python
sns.histplot(x='platelets',data=df ,bins=13,hue='DEATH_EVENT',kde='True' )
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9379006358>





![image-center](/assets/images/output_13_1.png){: .align-center}



```python
sns.jointplot(y='platelets',x='age',data=df)
```




    <seaborn.axisgrid.JointGrid at 0x7ff3add8ee48>





![image-center](/assets/images/output_14_1.png){: .align-center}





```python
sns.jointplot(x='platelets',y='creatinine_phosphokinase',hue='DEATH_EVENT',data=df,alpha=0.3)
# 알파값을 주면 겹친 부분이 진해져서 보임
```




    <seaborn.axisgrid.JointGrid at 0x7f93765dcc88>





![image-center](/assets/images/output_15_1.png){: .align-center}




```python
df.columns
```




    Index(['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
           'ejection_fraction', 'high_blood_pressure', 'platelets',
           'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time',
           'DEATH_EVENT'],
          dtype='object')



### Boxplot 계열을 이용하여 범주별 통계 확인하기



```python
# hue 키워드를 사용하여 범주 세분화 가능
# ejection_fraction 는 사망에 큰 영향이 없음
sns.boxplot(data=df,x='DEATH_EVENT',y='ejection_fraction') # 0이 생존
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7ff3a4b4c780>





![image-center](/assets/images/output_18_1.png){: .align-center}



```python
# ejection_fraction는 흡연 여부와 크게 상관 없음!
sns.boxplot(data=df,x='smoking',y='ejection_fraction')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f93764451d0>





![image-center](/assets/images/output_19_1.png){: .align-center}





```python
sns.violinplot(data=df,x='DEATH_EVENT',y='ejection_fraction')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f93763602e8>





![image-center](/assets/images/output_20_1.png){: .align-center}



```python
sns.swarmplot(data=df,x='DEATH_EVENT',y='platelets',hue='smoking')
```

    /usr/local/lib/python3.6/dist-packages/seaborn/categorical.py:1296: UserWarning: 9.9% of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.
      warnings.warn(msg, UserWarning)





    <matplotlib.axes._subplots.AxesSubplot at 0x7f93762a0f28>



![image-center](/assets/images/output_21_2.png){: .align-center}




```python
df.cov()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>anaemia</th>
      <th>creatinine_phosphokinase</th>
      <th>diabetes</th>
      <th>ejection_fraction</th>
      <th>high_blood_pressure</th>
      <th>platelets</th>
      <th>serum_creatinine</th>
      <th>serum_sodium</th>
      <th>sex</th>
      <th>smoking</th>
      <th>time</th>
      <th>DEATH_EVENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>age</th>
      <td>141.486483</td>
      <td>0.519335</td>
      <td>-9.415915e+02</td>
      <td>-0.593633</td>
      <td>8.460237</td>
      <td>0.530565</td>
      <td>-6.090712e+04</td>
      <td>1.958845</td>
      <td>-2.412544</td>
      <td>0.372120</td>
      <td>0.103847</td>
      <td>-206.861351</td>
      <td>1.411454</td>
    </tr>
    <tr>
      <th>anaemia</th>
      <td>0.519335</td>
      <td>0.246122</td>
      <td>-9.181641e+01</td>
      <td>-0.003120</td>
      <td>0.185282</td>
      <td>0.009057</td>
      <td>-2.124536e+03</td>
      <td>0.026777</td>
      <td>0.091681</td>
      <td>-0.022480</td>
      <td>-0.024893</td>
      <td>-5.445142</td>
      <td>0.015376</td>
    </tr>
    <tr>
      <th>creatinine_phosphokinase</th>
      <td>-941.591531</td>
      <td>-91.816413</td>
      <td>9.414586e+05</td>
      <td>-4.620581</td>
      <td>-506.174452</td>
      <td>-32.748805</td>
      <td>2.321533e+06</td>
      <td>-16.470382</td>
      <td>254.956443</td>
      <td>37.017261</td>
      <td>1.098696</td>
      <td>-703.803618</td>
      <td>28.464468</td>
    </tr>
    <tr>
      <th>diabetes</th>
      <td>-0.593633</td>
      <td>-0.003120</td>
      <td>-4.620581e+00</td>
      <td>0.244102</td>
      <td>-0.028361</td>
      <td>-0.003008</td>
      <td>4.454928e+03</td>
      <td>-0.024010</td>
      <td>-0.195226</td>
      <td>-0.037261</td>
      <td>-0.034006</td>
      <td>1.293259</td>
      <td>-0.000449</td>
    </tr>
    <tr>
      <th>ejection_fraction</th>
      <td>8.460237</td>
      <td>0.185282</td>
      <td>-5.061745e+02</td>
      <td>-0.028361</td>
      <td>140.063455</td>
      <td>0.138325</td>
      <td>8.354524e+04</td>
      <td>-0.138379</td>
      <td>9.185787</td>
      <td>-0.839667</td>
      <td>-0.372573</td>
      <td>38.330464</td>
      <td>-1.486667</td>
    </tr>
    <tr>
      <th>high_blood_pressure</th>
      <td>0.530565</td>
      <td>0.009057</td>
      <td>-3.274880e+01</td>
      <td>-0.003008</td>
      <td>0.138325</td>
      <td>0.228614</td>
      <td>2.336480e+03</td>
      <td>-0.002441</td>
      <td>0.078292</td>
      <td>-0.023916</td>
      <td>-0.012458</td>
      <td>-7.289904</td>
      <td>0.017744</td>
    </tr>
    <tr>
      <th>platelets</th>
      <td>-60907.118586</td>
      <td>-2124.535856</td>
      <td>2.321533e+06</td>
      <td>4454.928228</td>
      <td>83545.241001</td>
      <td>2336.480427</td>
      <td>9.565669e+09</td>
      <td>-4168.399498</td>
      <td>26810.436905</td>
      <td>-5851.104689</td>
      <td>1291.447854</td>
      <td>79811.066099</td>
      <td>-2247.619159</td>
    </tr>
    <tr>
      <th>serum_creatinine</th>
      <td>1.958845</td>
      <td>0.026777</td>
      <td>-1.647038e+01</td>
      <td>-0.024010</td>
      <td>-0.138379</td>
      <td>-0.002441</td>
      <td>-4.168399e+03</td>
      <td>1.070211</td>
      <td>-0.863173</td>
      <td>0.003448</td>
      <td>-0.013263</td>
      <td>-11.988935</td>
      <td>0.142374</td>
    </tr>
    <tr>
      <th>serum_sodium</th>
      <td>-2.412544</td>
      <td>0.091681</td>
      <td>2.549564e+02</td>
      <td>-0.195226</td>
      <td>9.185787</td>
      <td>0.078292</td>
      <td>2.681044e+04</td>
      <td>-0.863173</td>
      <td>19.469956</td>
      <td>-0.058158</td>
      <td>0.009932</td>
      <td>30.014152</td>
      <td>-0.402819</td>
    </tr>
    <tr>
      <th>sex</th>
      <td>0.372120</td>
      <td>-0.022480</td>
      <td>3.701726e+01</td>
      <td>-0.037261</td>
      <td>-0.839667</td>
      <td>-0.023916</td>
      <td>-5.851105e+03</td>
      <td>0.003448</td>
      <td>-0.058158</td>
      <td>0.228614</td>
      <td>0.099706</td>
      <td>-0.579224</td>
      <td>-0.000965</td>
    </tr>
    <tr>
      <th>smoking</th>
      <td>0.103847</td>
      <td>-0.024893</td>
      <td>1.098696e+00</td>
      <td>-0.034006</td>
      <td>-0.372573</td>
      <td>-0.012458</td>
      <td>1.291448e+03</td>
      <td>-0.013263</td>
      <td>0.009932</td>
      <td>0.099706</td>
      <td>0.218716</td>
      <td>-0.829005</td>
      <td>-0.002761</td>
    </tr>
    <tr>
      <th>time</th>
      <td>-206.861351</td>
      <td>-5.445142</td>
      <td>-7.038036e+02</td>
      <td>1.293259</td>
      <td>38.330464</td>
      <td>-7.289904</td>
      <td>7.981107e+04</td>
      <td>-11.988935</td>
      <td>30.014152</td>
      <td>-0.579224</td>
      <td>-0.829005</td>
      <td>6023.965276</td>
      <td>-19.127663</td>
    </tr>
    <tr>
      <th>DEATH_EVENT</th>
      <td>1.411454</td>
      <td>0.015376</td>
      <td>2.846447e+01</td>
      <td>-0.000449</td>
      <td>-1.486667</td>
      <td>0.017744</td>
      <td>-2.247619e+03</td>
      <td>0.142374</td>
      <td>-0.402819</td>
      <td>-0.000965</td>
      <td>-0.002761</td>
      <td>-19.127663</td>
      <td>0.218716</td>
    </tr>
  </tbody>
</table>

</div>



## Step 3. 모델 학습을 위한 데이터 전처리



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>anaemia</th>
      <th>creatinine_phosphokinase</th>
      <th>diabetes</th>
      <th>ejection_fraction</th>
      <th>high_blood_pressure</th>
      <th>platelets</th>
      <th>serum_creatinine</th>
      <th>serum_sodium</th>
      <th>sex</th>
      <th>smoking</th>
      <th>time</th>
      <th>DEATH_EVENT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>75.0</td>
      <td>0</td>
      <td>582</td>
      <td>0</td>
      <td>20</td>
      <td>1</td>
      <td>265000.00</td>
      <td>1.9</td>
      <td>130</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>55.0</td>
      <td>0</td>
      <td>7861</td>
      <td>0</td>
      <td>38</td>
      <td>0</td>
      <td>263358.03</td>
      <td>1.1</td>
      <td>136</td>
      <td>1</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>65.0</td>
      <td>0</td>
      <td>146</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>162000.00</td>
      <td>1.3</td>
      <td>129</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.0</td>
      <td>1</td>
      <td>111</td>
      <td>0</td>
      <td>20</td>
      <td>0</td>
      <td>210000.00</td>
      <td>1.9</td>
      <td>137</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>65.0</td>
      <td>1</td>
      <td>160</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>327000.00</td>
      <td>2.7</td>
      <td>116</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
    </tr>
  </tbody>
</table>

</div>



### StandardScaler를 이용하여 데이터 전처리하기



```python
from sklearn.preprocessing import StandardScaler
```


```python
# 수치형 입력 데이터, 범주형 입력 데이터, 출력 데이터로 구분하기
X_num = df[['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time']]
X_cat = df[['anaemia','diabetes','high_blood_pressure','sex','smoking']]
y =  df['DEATH_EVENT']


scaler = StandardScaler() 
scaler.fit(X_num)
X_scaled = scaler.transform(X_num) # numpy로 변환 # index,col 정보가 사라짐
X_scaled = pd.DataFrame(data=X_scaled,columns=X_num.columns,index=X_num.index)

x = pd.concat([X_scaled,X_cat],axis=1)


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
```


```python
# 수치형 입력 데이터를 전처리하고 입력 데이터 통합하기
scaler = StandardScaler() 
scaler.fit(X_num)
X_scaled = scaler.transform(X_num) # numpy로 변환 # index,col 정보가 사라짐
X_scaled = pd.DataFrame(data=X_scaled,columns=X_num.columns,index=X_num.index)

x = pd.concat([X_scaled,X_cat],axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }


    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>creatinine_phosphokinase</th>
      <th>ejection_fraction</th>
      <th>platelets</th>
      <th>serum_creatinine</th>
      <th>serum_sodium</th>
      <th>time</th>
      <th>anaemia</th>
      <th>diabetes</th>
      <th>high_blood_pressure</th>
      <th>sex</th>
      <th>smoking</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.192945</td>
      <td>0.000166</td>
      <td>-1.530560</td>
      <td>1.681648e-02</td>
      <td>0.490057</td>
      <td>-1.504036</td>
      <td>-1.629502</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.491279</td>
      <td>7.514640</td>
      <td>-0.007077</td>
      <td>7.535660e-09</td>
      <td>-0.284552</td>
      <td>-0.141976</td>
      <td>-1.603691</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.350833</td>
      <td>-0.449939</td>
      <td>-1.530560</td>
      <td>-1.038073e+00</td>
      <td>-0.090900</td>
      <td>-1.731046</td>
      <td>-1.590785</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.912335</td>
      <td>-0.486071</td>
      <td>-1.530560</td>
      <td>-5.464741e-01</td>
      <td>0.490057</td>
      <td>0.085034</td>
      <td>-1.590785</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.350833</td>
      <td>-0.435486</td>
      <td>-1.530560</td>
      <td>6.517986e-01</td>
      <td>1.264666</td>
      <td>-4.682176</td>
      <td>-1.577879</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

</div>



###  학습데이터와 테스트데이터 분리하기



```python
from sklearn.model_selection import train_test_split
```


```python
# train_test_split() 함수로 학습 데이터와 테스트 데이터 분리하기
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
```

## Step 4. Classification 모델 학습하기


###  Logistic Regression 모델 생성/학습하기



```python
from sklearn.linear_model import LogisticRegression
```


```python
# LogisticRegression 모델 생성/학습
model_lr = LogisticRegression()
model_lr.fit(X_train,y_train)
```

###  모델 학습 결과 평가하기



```python
from sklearn.metrics import classification_report
```


```python
# Predict를 수행하고 classification_report() 결과 출력하기
pred = model_lr.predict(X_test)
print(classification_report(y_test, pred))
```

                  precision    recall  f1-score   support
    
               0       0.86      0.92      0.89        64
               1       0.76      0.62      0.68        26
    
        accuracy                           0.83        90
       macro avg       0.81      0.77      0.78        90
    weighted avg       0.83      0.83      0.83        90


​    

###  XGBoost 모델 생성/학습하기



```python
from xgboost import XGBClassifier
```


```python
# XGBClassifier 모델 생성/학습
model_xgb = XGBClassifier()
model_xgb.fit(X_train,y_train)



pred = model_xgb.predict(X_test)

print(classification_report(y_test, pred))
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0,
                  learning_rate=0.1, max_delta_step=0, max_depth=3,
                  min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
                  nthread=None, objective='binary:logistic', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                  silent=None, subsample=1, verbosity=1)



###  모델 학습 결과 평가하기



```python
# Predict를 수행하고 classification_report() 결과 출력하기
pred = model_xgb.predict(X_test)

print(classification_report(y_test, pred))
# xgb 모델의 정확성이  LogisticRegression 모델보다 높음
```

                  precision    recall  f1-score   support
    
               0       0.90      0.97      0.93        64
               1       0.90      0.73      0.81        26
    
        accuracy                           0.90        90
       macro avg       0.90      0.85      0.87        90
    weighted avg       0.90      0.90      0.90        90


​    

###  특징의 중요도 확인하기



```python
# XGBClassifier 모델의 feature_importances_를 이용하여 중요도 plot

plt.bar(x.columns,model_xgb.feature_importances_)
plt.xticks(rotation=90)
plt.show()

# time이 중요요소?
# time의 관찰일의 폭이 너무 넓음 다양하게 존재
# 제거하고 해보기

#   age: 환자의 나이
#   ejection_fraction: 박출계수 (%)
#   serum_creatinine: 혈중 크레아틴 레벨 (mg/dL)
#   smoking: 흡연 여부 (0: 비흡연, 1: 흡연)
#  위의 4가지 항목들이 심부전증에 많은 영향을 줌
```


![png](output_45_0.png)



```python
# time의 관찰일의 폭이 너무 넓음 다양하게 존재
sns.histplot(x='time',data=df,hue='DEATH_EVENT',kde=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f936b0b6cc0>



![image-center](/assets/images/output_46_1.png){: .align-center}




```python
# time을 제거하고 
X_num = df[['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium']]
X_cat = df[['anaemia','diabetes','high_blood_pressure','sex','smoking']]
y =  df['DEATH_EVENT']


scaler = StandardScaler() 
scaler.fit(X_num)
X_scaled = scaler.transform(X_num) # numpy로 변환 # index,col 정보가 사라짐
X_scaled = pd.DataFrame(data=X_scaled,columns=X_num.columns,index=X_num.index)

x = pd.concat([X_scaled,X_cat],axis=1)


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
pred = model_lr.predict(X_test)
print(classification_report(y_test, pred))

model_xgb = XGBClassifier()
model_xgb.fit(X_train,y_train)

pred = model_xgb.predict(X_test)
print(classification_report(y_test, pred))
# 제거하니 xgb 모델의 정확성이 80% 까지 떨어짐
```

                  precision    recall  f1-score   support
    
               0       0.78      0.92      0.84        64
               1       0.64      0.35      0.45        26
    
        accuracy                           0.76        90
       macro avg       0.71      0.63      0.65        90
    weighted avg       0.74      0.76      0.73        90
    
                  precision    recall  f1-score   support
    
               0       0.84      0.89      0.86        64
               1       0.68      0.58      0.62        26
    
        accuracy                           0.80        90
       macro avg       0.76      0.73      0.74        90
    weighted avg       0.79      0.80      0.79        90


​    


```python
plt.bar(x.columns,model_xgb.feature_importances_)
plt.xticks(rotation=90)
plt.show()
# time이 빠지고 다른특징의 중요도
```


![png](output_48_0.png)



```python
sns.jointplot(data=df,x='ejection_fraction',y='serum_creatinine',hue='DEATH_EVENT')
# 사망의 분류가 잘됨
```




    <seaborn.axisgrid.JointGrid at 0x7f936aebb390>





![image-center](/assets/images/output_49_1.png){: .align-center}



## Step5 모델 학습 결과 심화 분석하기


###  Precision-Recall 커브 확인하기

 - True와 False가 서로 불균형하게 있는 경우에 어떻게 성능평가를 제대로 할 수 있는지에 확인하기 위해서 Precision과 Recall을 이용해야 한다.



```python
from sklearn.metrics import plot_precision_recall_curve
```


```python
# 두 모델의 Precision-Recall 커브를 한번에 그리기 ! fig.gca()로 ax를 반환받아 사용

fig = plt.figure()
ax = fig.gca()
plot_precision_recall_curve(model_lr, X_test, y_test, ax=ax)
plot_precision_recall_curve(model_xgb, X_test, y_test, ax=ax)
```




    <sklearn.metrics._plot.precision_recall_curve.PrecisionRecallDisplay at 0x7f936ad574a8>






![image-center](/assets/images/output_53_1.png){: .align-center}



###  ROC 커브 확인하기

-  두 클래스를 더 잘 구별할 수 있다면 ROC 커브는 좌상단에 더 가까워지게 된다.



```python
from sklearn.metrics import plot_roc_curve
```


```python
fig = plt.figure()
ax = fig.gca()
plot_roc_curve(model_lr, X_test, y_test, ax=ax)
plot_roc_curve(model_xgb, X_test, y_test, ax=ax)
```




    <sklearn.metrics._plot.roc_curve.RocCurveDisplay at 0x7f936acc0b70>






![image-center](/assets/images/output_56_1.png){: .align-center}



### 스케일 작업 없이


```python
'''
원본 데이터는 데이터 고유의 특성과 분포를 가지고 있다
이를 그대로 사용하게 되면 학습이 느리거나 문제가 발생하는 경우가 종종 발생하여
Scaler를 이용하여 동일하게 일정 범위로 스케일링하는 것이 필요
밑에 결과를 보면 스케일러를 사용할 떄와 결과값이 다른것을 알 수 있음

'''


X_num = df[['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium']]
X_cat = df[['anaemia','diabetes','high_blood_pressure','sex','smoking']]
y =  df['DEATH_EVENT']


x = pd.concat([X_num,X_cat],axis=1)


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
pred = model_lr.predict(X_test)
print(classification_report(y_test, pred))

model_xgb = XGBClassifier()
model_xgb.fit(X_train,y_train)

pred = model_xgb.predict(X_test)
print(classification_report(y_test, pred))


```

                  precision    recall  f1-score   support
    
               0       0.77      0.94      0.85        64
               1       0.67      0.31      0.42        26
    
        accuracy                           0.76        90
       macro avg       0.72      0.62      0.63        90
    weighted avg       0.74      0.76      0.72        90
    
                  precision    recall  f1-score   support
    
               0       0.84      0.89      0.86        64
               1       0.68      0.58      0.62        26
    
        accuracy                           0.80        90
       macro avg       0.76      0.73      0.74        90
    weighted avg       0.79      0.80      0.79        90


​    


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
# time을 제거하고 
X_num = df[['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium']]
X_cat = df[['anaemia','diabetes','high_blood_pressure','sex','smoking']]
y =  df['DEATH_EVENT']


scaler = StandardScaler() 
scaler.fit(X_num)
X_scaled = scaler.transform(X_num) # numpy로 변환 # index,col 정보가 사라짐
X_scaled = pd.DataFrame(data=X_scaled,columns=X_num.columns,index=X_num.index)

x = pd.concat([X_scaled,X_cat],axis=1)


X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)

# 랜덤포레스트 모델
clf = RandomForestClassifier(n_estimators = 500,max_depth=15,
                             random_state=0) 
clf.fit(X_train,y_train)

pred =clf.predict(X_test)
print(classification_report(y_test, pred))
```

                  precision    recall  f1-score   support
    
               0       0.80      0.89      0.84        64
               1       0.63      0.46      0.53        26
    
        accuracy                           0.77        90
       macro avg       0.72      0.68      0.69        90
    weighted avg       0.75      0.77      0.75        90


​    
