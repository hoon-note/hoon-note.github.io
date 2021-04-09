---
layout: single
title: 'Python ML 연습문제(학습 요소 분석하기)'
---

#  학습 성공/실패 요소 분석해보기


    gender: 학생의 성별 (M: 남성, F: 여성)
    NationaliTy: 학생의 국적
    PlaceofBirth: 학생이 태어난 국가
    StageID: 학생이 다니는 학교 (초,중,고)
    GradeID: 학생이 속한 성적 등급
    SectionID: 학생이 속한 반 이름
    Topic: 수강한 과목
    Semester: 수강한 학기 (1학기/2학기)
    Relation: 주 보호자와 학생의 관계
    raisedhands: 학생이 수업 중 손을 든 횟수
    VisITedResources: 학생이 과목 공지를 확인한 횟수
    Discussion: 학생이 토론 그룹에 참여한 횟수
    ParentAnsweringSurvey: 부모가 학교 설문에 참여했는지 여부
    ParentschoolSatisfaction: 부모가 학교에 만족했는지 여부
    StudentAbscenceDays: 학생의 결석 횟수 (7회 이상/미만)
    Class: 학생의 성적 등급 (L: 낮음, M: 보통, H: 높음)


​    
​    

- 데이터 출처: https://www.kaggle.com/aljarah/xAPI-Edu-Data

---

## Step 1. 데이터셋 준비하기


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
# pd.read_csv()로 csv파일 읽어들이기
df = pd.read_csv('xAPI-Edu-Data.csv')
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
      <th>gender</th>
      <th>NationalITy</th>
      <th>PlaceofBirth</th>
      <th>StageID</th>
      <th>GradeID</th>
      <th>SectionID</th>
      <th>Topic</th>
      <th>Semester</th>
      <th>Relation</th>
      <th>raisedhands</th>
      <th>VisITedResources</th>
      <th>AnnouncementsView</th>
      <th>Discussion</th>
      <th>ParentAnsweringSurvey</th>
      <th>ParentschoolSatisfaction</th>
      <th>StudentAbsenceDays</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M</td>
      <td>KW</td>
      <td>KuwaIT</td>
      <td>lowerlevel</td>
      <td>G-04</td>
      <td>A</td>
      <td>IT</td>
      <td>F</td>
      <td>Father</td>
      <td>15</td>
      <td>16</td>
      <td>2</td>
      <td>20</td>
      <td>Yes</td>
      <td>Good</td>
      <td>Under-7</td>
      <td>M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M</td>
      <td>KW</td>
      <td>KuwaIT</td>
      <td>lowerlevel</td>
      <td>G-04</td>
      <td>A</td>
      <td>IT</td>
      <td>F</td>
      <td>Father</td>
      <td>20</td>
      <td>20</td>
      <td>3</td>
      <td>25</td>
      <td>Yes</td>
      <td>Good</td>
      <td>Under-7</td>
      <td>M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M</td>
      <td>KW</td>
      <td>KuwaIT</td>
      <td>lowerlevel</td>
      <td>G-04</td>
      <td>A</td>
      <td>IT</td>
      <td>F</td>
      <td>Father</td>
      <td>10</td>
      <td>7</td>
      <td>0</td>
      <td>30</td>
      <td>No</td>
      <td>Bad</td>
      <td>Above-7</td>
      <td>L</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M</td>
      <td>KW</td>
      <td>KuwaIT</td>
      <td>lowerlevel</td>
      <td>G-04</td>
      <td>A</td>
      <td>IT</td>
      <td>F</td>
      <td>Father</td>
      <td>30</td>
      <td>25</td>
      <td>5</td>
      <td>35</td>
      <td>No</td>
      <td>Bad</td>
      <td>Above-7</td>
      <td>L</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M</td>
      <td>KW</td>
      <td>KuwaIT</td>
      <td>lowerlevel</td>
      <td>G-04</td>
      <td>A</td>
      <td>IT</td>
      <td>F</td>
      <td>Father</td>
      <td>40</td>
      <td>50</td>
      <td>12</td>
      <td>50</td>
      <td>No</td>
      <td>Bad</td>
      <td>Above-7</td>
      <td>M</td>
    </tr>
  </tbody>
</table>

</div>



## Step 2. EDA 및 데이터 기초 통계 분석


### 데이터프레임의 각 컬럼 분석하기



```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 480 entries, 0 to 479
    Data columns (total 17 columns):
     #   Column                    Non-Null Count  Dtype 
    ---  ------                    --------------  ----- 
     0   gender                    480 non-null    object
     1   NationalITy               480 non-null    object
     2   PlaceofBirth              480 non-null    object
     3   StageID                   480 non-null    object
     4   GradeID                   480 non-null    object
     5   SectionID                 480 non-null    object
     6   Topic                     480 non-null    object
     7   Semester                  480 non-null    object
     8   Relation                  480 non-null    object
     9   raisedhands               480 non-null    int64 
     10  VisITedResources          480 non-null    int64 
     11  AnnouncementsView         480 non-null    int64 
     12  Discussion                480 non-null    int64 
     13  ParentAnsweringSurvey     480 non-null    object
     14  ParentschoolSatisfaction  480 non-null    object
     15  StudentAbsenceDays        480 non-null    object
     16  Class                     480 non-null    object
    dtypes: int64(4), object(13)
    memory usage: 63.9+ KB



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
      <th>raisedhands</th>
      <th>VisITedResources</th>
      <th>AnnouncementsView</th>
      <th>Discussion</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
      <td>480.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>46.775000</td>
      <td>54.797917</td>
      <td>37.918750</td>
      <td>43.283333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>30.779223</td>
      <td>33.080007</td>
      <td>26.611244</td>
      <td>27.637735</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>15.750000</td>
      <td>20.000000</td>
      <td>14.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>50.000000</td>
      <td>65.000000</td>
      <td>33.000000</td>
      <td>39.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>75.000000</td>
      <td>84.000000</td>
      <td>58.000000</td>
      <td>70.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.000000</td>
      <td>99.000000</td>
      <td>98.000000</td>
      <td>99.000000</td>
    </tr>
  </tbody>
</table>

</div>




```python
df['gender'].value_counts()
```




    M    305
    F    175
    Name: gender, dtype: int64




```python
df['NationalITy'].value_counts()
```




    KW             179
    Jordan         172
    Palestine       28
    Iraq            22
    lebanon         17
    Tunis           12
    SaudiArabia     11
    Egypt            9
    Syria            7
    Lybia            6
    Iran             6
    USA              6
    Morocco          4
    venzuela         1
    Name: NationalITy, dtype: int64




```python
df['PlaceofBirth'].value_counts()
```




    KuwaIT         180
    Jordan         176
    Iraq            22
    lebanon         19
    USA             16
    SaudiArabia     16
    Palestine       10
    Egypt            9
    Tunis            9
    Syria            6
    Lybia            6
    Iran             6
    Morocco          4
    venzuela         1
    Name: PlaceofBirth, dtype: int64




```python
df.columns
```




    Index(['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
           'SectionID', 'Topic', 'Semester', 'Relation', 'raisedhands',
           'VisITedResources', 'AnnouncementsView', 'Discussion',
           'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
           'StudentAbsenceDays', 'Class'],
          dtype='object')



###  수치형 데이터의 히스토그램 그리기



```python
sns.histplot(data=df,x='raisedhands',hue='Class',hue_order=['L','M','H'],kde=True)

# 손을 많이 든 학생이 성적이 높은 클래스인것을 알 수 있음 
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7faa7bec9400>



![image-center](/assets/images/output_14_1.png){: .align-center}




```python
sns.histplot(data=df,x='VisITedResources',hue='Class',hue_order=['L','M','H'],kde=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7faa7b366ef0>



![image-center](/assets/images/output_15_1.png){: .align-center}




```python
sns.histplot(data=df,x='AnnouncementsView',hue='Class',hue_order=['L','M','H'],kde=True)
# 성적이 좋은것은 공지랑은 크게 상관이 없다고 봄
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7faa7ae0bd30>



![image-center](/assets/images/output_16_1.png){: .align-center}



```python
sns.histplot(data=df,x='Discussion',hue='Class',hue_order=['L','M','H'],kde=True)

# 상관성이 좋지가 않음 생각보다
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7faa7adedef0>



![image-center](/assets/images/output_17_1.png){: .align-center}



```python
sns.jointplot(data=df,x='VisITedResources',y='raisedhands',hue='Class',hue_order=['L','M','H'])
```




    <seaborn.axisgrid.JointGrid at 0x7faa7ac80b70>



![image-center](/assets/images/output_18_1.png){: .align-center}



```python
sns.pairplot(data=df,hue='Class',hue_order=['L','M','H'])
```




    <seaborn.axisgrid.PairGrid at 0x7faa7ad9c748>



![image-center](/assets/images/output_19_1.png){: .align-center}




###  Countplot을 이용하여 범주별 통계 확인하기



```python
'''
 gender: 학생의 성별 (M: 남성, F: 여성)
    NationaliTy: 학생의 국적
    PlaceofBirth: 학생이 태어난 국가
    StageID: 학생이 다니는 학교 (초,중,고)
    GradeID: 학생이 속한 성적 등급
    SectionID: 학생이 속한 반 이름
    Topic: 수강한 과목
    Semester: 수강한 학기 (1학기/2학기)
    Relation: 주 보호자와 학생의 관계
    raisedhands: 학생이 수업 중 손을 든 횟수
    VisITedResources: 학생이 과목 공지를 확인한 횟수
    Discussion: 학생이 토론 그룹에 참여한 횟수
    ParentAnsweringSurvey: 부모가 학교 설문에 참여했는지 여부
    ParentschoolSatisfaction: 부모가 학교에 만족했는지 여부
    StudentAbscenceDays: 학생의 결석 횟수 (7회 이상/미만)
    Class: 학생의 성적 등급 (L: 낮음, M: 보통, H: 높음)

'''
df.columns
```




    Index(['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
           'SectionID', 'Topic', 'Semester', 'Relation', 'raisedhands',
           'VisITedResources', 'AnnouncementsView', 'Discussion',
           'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
           'StudentAbsenceDays', 'Class'],
          dtype='object')




```python
# seaborn의 countplot()을 사용
# x와 hue를 사용하여 범주별 Class 통계 확인

sns.countplot(data=df,x='gender',hue='Class',hue_order=['L','M','H'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7faa7048bb00>



![image-center](/assets/images/output_22_1.png){: .align-center}




```python
sns.countplot(data=df,x='ParentschoolSatisfaction',hue='Class',hue_order=['L','M','H'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7faa703db860>





![image-center](/assets/images/output_23_1.png){: .align-center}




```python
sns.countplot(data=df,x='NationalITy',hue='Class',hue_order=['L','M','H'])
plt.xticks(rotation=90)
plt.show()
```



![image-center](/assets/images/output_24_0.png){: .align-center}



```python
sns.countplot(data=df,x='StudentAbsenceDays',hue='Class',hue_order=['L','M','H'])
# 결석 특징은 성적과 밀접한 상관성이 있다고 봄
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7faa703935c0>





![image-center](/assets/images/output_25_1.png){: .align-center}



```python
sns.countplot(data=df,x='ParentAnsweringSurvey',hue='Class',hue_order=['L','M','H'])
# 부모가 서베이에 응했을때 성적이 낮은 학생의 비율이 낮다고 생각함
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7faa70220860>





![image-center](/assets/images/output_26_1.png){: .align-center}



```python
sns.countplot(data=df,x='Topic',hue='Class',hue_order=['L','M','H'])
plt.xticks(rotation=90)
plt.show()
# it 과목이 성적이 낮게 나오는걸 알 수 있음 (학생들이 어떤 과목을 어려워하는지 알 수 있음)
```




![image-center](/assets/images/output_27_0.png){: .align-center}


###  범주형 대상 Class 컬럼을 수치로 바꾸어 표현하기


```python
# L, M, H를 숫자로 바꾸어 표현하기 (eg. L: -1, M: 0, H:1)
# Hint) DataFrame의 map() 메소드를 사용
grade = {
    'L':-1,
    'M':0,
    'H':1
}
df['Class_value'] = df['Class'].map(grade)
```


```python
# 숫자로 바꾼 Class_value 컬럼을 이용해 다양한 시각화 수행하기

gb_gender = df.groupby('gender').mean()['Class_value']
plt.bar(gb_gender.index,gb_gender)

# 남학생은 성적이 안좋은 학생들이 더 많았음
```




    <BarContainer object of 2 artists>




![image-center](/assets/images/output_30_1.png){: .align-center}



```python
gb_Topic = df.groupby('Topic').mean()['Class_value'].sort_values()
plt.barh(gb_Topic.index,gb_Topic)

# 스페인어와 it를 선택한 학생들이 어려워함
```




    <BarContainer object of 12 artists>




![image-center](/assets/images/output_31_1.png){: .align-center}


```python
gb_StudentAbsenceDays = df.groupby('StudentAbsenceDays').mean()['Class_value'].sort_values(ascending=False)
plt.barh(gb_StudentAbsenceDays.index,gb_StudentAbsenceDays)

# 결석일수가 성적에 영향을 끼침
```




    <BarContainer object of 2 artists>





![image-center](/assets/images/output_32_1.png){: .align-center}



## Step 3. 모델 학습을 위한 데이터 전처리


###  get_dummies()를 이용하여 범주형 데이터 전처리하기



```python
df.columns
```




    Index(['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
           'SectionID', 'Topic', 'Semester', 'Relation', 'raisedhands',
           'VisITedResources', 'AnnouncementsView', 'Discussion',
           'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
           'StudentAbsenceDays', 'Class', 'Class_value'],
          dtype='object')




```python
# pd.get_dummies()를 이용해 범주형 데이터를 one-hot 벡터로 변환하기
# Multicollinearity를 피하기 위해 drop_first=True로 설정

X = pd.get_dummies(df.drop(['ParentschoolSatisfaction','Class','Class_value'],axis=1) # 사용안하는 데이터는 drop
                   ,columns=[ 'gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID',
                                'SectionID', 'Topic', 'Semester', 'Relation', 
                                 'ParentAnsweringSurvey',
                                'StudentAbsenceDays'],drop_first=True)
y = df['Class']
```


```python
X
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
      <th>raisedhands</th>
      <th>VisITedResources</th>
      <th>AnnouncementsView</th>
      <th>Discussion</th>
      <th>gender_M</th>
      <th>NationalITy_Iran</th>
      <th>NationalITy_Iraq</th>
      <th>NationalITy_Jordan</th>
      <th>NationalITy_KW</th>
      <th>NationalITy_Lybia</th>
      <th>NationalITy_Morocco</th>
      <th>NationalITy_Palestine</th>
      <th>NationalITy_SaudiArabia</th>
      <th>NationalITy_Syria</th>
      <th>NationalITy_Tunis</th>
      <th>NationalITy_USA</th>
      <th>NationalITy_lebanon</th>
      <th>NationalITy_venzuela</th>
      <th>PlaceofBirth_Iran</th>
      <th>PlaceofBirth_Iraq</th>
      <th>PlaceofBirth_Jordan</th>
      <th>PlaceofBirth_KuwaIT</th>
      <th>PlaceofBirth_Lybia</th>
      <th>PlaceofBirth_Morocco</th>
      <th>PlaceofBirth_Palestine</th>
      <th>PlaceofBirth_SaudiArabia</th>
      <th>PlaceofBirth_Syria</th>
      <th>PlaceofBirth_Tunis</th>
      <th>PlaceofBirth_USA</th>
      <th>PlaceofBirth_lebanon</th>
      <th>PlaceofBirth_venzuela</th>
      <th>StageID_MiddleSchool</th>
      <th>StageID_lowerlevel</th>
      <th>GradeID_G-04</th>
      <th>GradeID_G-05</th>
      <th>GradeID_G-06</th>
      <th>GradeID_G-07</th>
      <th>GradeID_G-08</th>
      <th>GradeID_G-09</th>
      <th>GradeID_G-10</th>
      <th>GradeID_G-11</th>
      <th>GradeID_G-12</th>
      <th>SectionID_B</th>
      <th>SectionID_C</th>
      <th>Topic_Biology</th>
      <th>Topic_Chemistry</th>
      <th>Topic_English</th>
      <th>Topic_French</th>
      <th>Topic_Geology</th>
      <th>Topic_History</th>
      <th>Topic_IT</th>
      <th>Topic_Math</th>
      <th>Topic_Quran</th>
      <th>Topic_Science</th>
      <th>Topic_Spanish</th>
      <th>Semester_S</th>
      <th>Relation_Mum</th>
      <th>ParentAnsweringSurvey_Yes</th>
      <th>StudentAbsenceDays_Under-7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15</td>
      <td>16</td>
      <td>2</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>20</td>
      <td>3</td>
      <td>25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>7</td>
      <td>0</td>
      <td>30</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>30</td>
      <td>25</td>
      <td>5</td>
      <td>35</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>50</td>
      <td>12</td>
      <td>50</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>475</th>
      <td>5</td>
      <td>4</td>
      <td>5</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>476</th>
      <td>50</td>
      <td>77</td>
      <td>14</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>477</th>
      <td>55</td>
      <td>74</td>
      <td>25</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>478</th>
      <td>30</td>
      <td>17</td>
      <td>14</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>479</th>
      <td>35</td>
      <td>14</td>
      <td>23</td>
      <td>62</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>480 rows × 59 columns</p>

</div>



###  학습데이터와 테스트데이터 분리하기



```python
from sklearn.model_selection import train_test_split
```


```python
# train_test_split() 함수로 학습 데이터와 테스트 데이터 분리하기
X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.3,random_state=1)
```




```python
len(X_train)
```




    336



## Step 4. Classification 모델 학습하기


### Logistic Regression 모델 생성/학습하기



```python
from sklearn.linear_model import LogisticRegression
```


```python
# LogisticRegression 모델 생성/학습
model_lr = LogisticRegression(max_iter=10000)
model_lr.fit(X_train,y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=10000,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)



### 모델 학습 결과 평가하기



```python
from sklearn.metrics import classification_report
```


```python
# Predict를 수행하고 classification_report() 결과 출력하기
pred = model_lr.predict(X_test)

print(classification_report(pred,y_test))
```

                  precision    recall  f1-score   support
    
               H       0.67      0.77      0.72        48
               L       0.76      0.78      0.77        32
               M       0.68      0.59      0.63        64
    
        accuracy                           0.69       144
       macro avg       0.70      0.72      0.71       144
    weighted avg       0.69      0.69      0.69       144


​    

###  XGBoost 모델 생성/학습하기



```python
from xgboost import XGBClassifier
```


```python
# XGBClassifier 모델 생성/학습
model_xgb = XGBClassifier( )
model_xgb.fit(X_train,y_train)
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=1, gamma=0,
                  learning_rate=0.1, max_delta_step=0, max_depth=3,
                  min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
                  nthread=None, objective='multi:softprob', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                  silent=None, subsample=1, verbosity=1)



###  모델 학습 결과 평가하기



```python
# Predict를 수행하고 classification_report() 결과 출력하기
pred = model_xgb.predict(X_test)

print(classification_report(pred,y_test))
```

                  precision    recall  f1-score   support
    
               H       0.64      0.83      0.72        42
               L       0.82      0.79      0.81        34
               M       0.75      0.62      0.68        68
    
        accuracy                           0.72       144
       macro avg       0.73      0.75      0.74       144
    weighted avg       0.73      0.72      0.72       144


​    

## Step5 모델 학습 결과 심화 분석하기


###  Logistic Regression 모델 계수로 상관성 파악하기


```python
model_lr.classes_
```




    array(['H', 'L', 'M'], dtype=object)




```python
model_lr.coef_.shape
```




    (3, 59)




```python
# Logistic Regression 모델의 coef_ 속성을 plot하기
# 성적에 영향을 주는 주요요소를 뽑아보기
fig = plt.figure(figsize=(15,8))
plt.bar(X.columns,model_lr.coef_[0,:])
plt.xticks(rotation=90)
plt.show()
# Logistic Regression 모델이 예측한 성적에 영향을 주는 요소들
# 책임자가 어머니인경우와 출결의 일수 ,서베이에 응한경우에 성적이 좋다고 봄
```


![image-center](/assets/images/output_59_0.png){: .align-center}



```python
fig = plt.figure(figsize=(15,8))
plt.bar(X.columns,model_lr.coef_[1,:])
plt.xticks(rotation=90)
plt.show()

# 성적이 안좋은데 기여하는 특징들
```


![image-center](/assets/images/output_60_0.png){: .align-center}


###  XGBoost 모델로 특징의 중요도 확인하기


```python
# XGBoost 모델의 feature_importances_ 속성을 plot하기

fig = plt.figure(figsize=(15,8))
plt.bar(X.columns,model_xgb.feature_importances_)
plt.xticks(rotation=90)
plt.show()

# 손을 많이 든 횟수도 중요도가 높아짐 
```


![image-center](/assets/images/output_62_0.png){: .align-center}


```python
from sklearn.ensemble import RandomForestClassifier


clf = RandomForestClassifier(n_estimators = 500,max_depth=15,
                            random_state=0) 
clf.fit(X_train,y_train)


pred =clf.predict(X_test)
print(classification_report(y_test, pred))
```

                  precision    recall  f1-score   support
    
               H       0.84      0.67      0.75        55
               L       0.82      0.85      0.84        33
               M       0.65      0.77      0.70        56
    
        accuracy                           0.75       144
       macro avg       0.77      0.76      0.76       144
    weighted avg       0.76      0.75      0.75       144


​    


```python
fig = plt.figure(figsize=(10,12))
plt.barh(X.columns,clf.feature_importances_)
plt.show()
```



![image-center](/assets/images/output_64_0.png){: .align-center}


