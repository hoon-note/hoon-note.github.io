---
layout: single
title: "Python ML Categorical_Variable"
---
#### 범주형 변수 처리

- 범주형변수는 상태공간의 크기가 유한한 변수
- 연속형변수는 상태공간의 크기가 무한한 변수
 - int혹은 float타입으로 정의된 변수는 반드시 연속형 변수가 아닐 수 있음
- 월,시간,초,분은 숫자지만 범주형 변수임 (다시 처음의 시간으로 돌아감) 유한한 변수
- 대부분의 데이터 분석 모형은 숫자만 입력으로 받을 수 있기 때문에 범주형 데이터는 숫자로 변환해야 한다. 범주형 데이터를 숫자로 변환하는 방법은 두가지다.
  - 더미변수화
  - 카테고리 임베딩


```python
import os
import pandas as pd
import numpy as np

```


```python
df = pd.read_csv("car-good.csv")
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
      <th>Buying</th>
      <th>Maint</th>
      <th>Doors</th>
      <th>Persons</th>
      <th>Lug_boot</th>
      <th>Safety</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>low</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>1</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>med</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>2</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>small</td>
      <td>high</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>3</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>med</td>
      <td>low</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>vhigh</td>
      <td>vhigh</td>
      <td>2</td>
      <td>2</td>
      <td>med</td>
      <td>med</td>
      <td>negative</td>
    </tr>
  </tbody>
</table>

</div>




```python
# 특징과 라벨 분리
X = df.drop('Class', axis = 1)
Y = df['Class']
```


```python
# 학습 데이터와 평가 데이터 분리
from sklearn.model_selection import train_test_split
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y)
```


```python
Train_Y.value_counts()
```




    negative    627
    positive     21
    Name: Class, dtype: int64




```python
# 문자 라벨을 숫자로 치환 
Train_Y.replace({"negative":-1, "positive":1}, inplace = True)
Test_Y.replace({"negative":-1, "positive":1}, inplace = True)
```


```python
Train_X.head() # Buying, Maint, Lug_boot, safety 변수가 범주형 변수로 판단됨
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
      <th>Buying</th>
      <th>Maint</th>
      <th>Doors</th>
      <th>Persons</th>
      <th>Lug_boot</th>
      <th>Safety</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>821</th>
      <td>low</td>
      <td>low</td>
      <td>2</td>
      <td>4</td>
      <td>small</td>
      <td>high</td>
    </tr>
    <tr>
      <th>406</th>
      <td>high</td>
      <td>low</td>
      <td>3</td>
      <td>4</td>
      <td>small</td>
      <td>med</td>
    </tr>
    <tr>
      <th>299</th>
      <td>high</td>
      <td>high</td>
      <td>3</td>
      <td>4</td>
      <td>small</td>
      <td>high</td>
    </tr>
    <tr>
      <th>148</th>
      <td>vhigh</td>
      <td>med</td>
      <td>4</td>
      <td>2</td>
      <td>med</td>
      <td>med</td>
    </tr>
    <tr>
      <th>840</th>
      <td>low</td>
      <td>low</td>
      <td>3</td>
      <td>4</td>
      <td>med</td>
      <td>low</td>
    </tr>
  </tbody>
</table>

</div>




```python
Train_Y.head()
```




    821    1
    406   -1
    299   -1
    148   -1
    840   -1
    Name: Class, dtype: int64




```python
# 자세한 범주형 변수 판별 => 모든 변수가 범주형임을 확인
for col in Train_X.columns:
    print(col, len(Train_X[col].unique()))
```

    Buying 4
    Maint 4
    Doors 3
    Persons 2
    Lug_boot 3
    Safety 3


#### 더미화를 이용한 범주 변수 처리 (상태 공간의 크기가 작을떄 사용)

- 차원이 늘어나고 희소해짐


```python
Train_X = Train_X.astype(str) # 모든 변수가 범주이므로, 더미화를 위해 전부 string 타입으로 변환
```


```python
from feature_engine.categorical_encoders import OneHotCategoricalEncoder as OHE
dummy_model = OHE(variables = Train_X.columns.tolist(), # 인덱스를 list를 바꿔줌 
                 drop_last = True)
# variables: 더미화 대상이 되는 범주형 변수 이름 목록(반드시 str타입이어야함,시리즈의 오브젝트타입)
# drop_last : 한 범주 변수로부터 만든 더미 변수 가운데 마지막 더미 변수를 제거할 지를 결정
# top_categories : 한 범주 변수로부터 만드는 더미 변수 개수를 설정하며 빈도 기준으로 자름
dummy_model.fit(Train_X)

d_Train_X = dummy_model.transform(Train_X)
d_Test_X = dummy_model.transform(Test_X)

'''
pandas의 get_dummies() 는 이 함수보다 간단하지만 학습데이터에 포함된 범주형 변수를 처리한 방식으로
새로 들어온 데이터에 적용이 불가능하기 떄문에, 실제적으로 활용이 어려움
'''
```


```python
# 더미화한 컬럼의 값들 
d_Train_X.head()
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
      <th>Buying_low</th>
      <th>Buying_high</th>
      <th>Buying_vhigh</th>
      <th>Maint_low</th>
      <th>Maint_high</th>
      <th>Maint_med</th>
      <th>Doors_2</th>
      <th>Doors_3</th>
      <th>Persons_4</th>
      <th>Lug_boot_small</th>
      <th>Lug_boot_med</th>
      <th>Safety_high</th>
      <th>Safety_med</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>821</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>406</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>299</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>148</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>840</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>

</div>




```python
# 더미화를 한 뒤의 모델 테스트
from sklearn.neighbors import KNeighborsClassifier as KNN
model = KNN().fit(d_Train_X, Train_Y) # knn 매트릭으로 자카드를 사용하는등의 성능향상을 할 수 있음
pred_Y = model.predict(d_Test_X)

from sklearn.metrics import f1_score
f1_score(Test_Y, pred_Y)
```




    0.0



#### 범주형 변수를 연속형 변수로 치환 (상태공간의 크키가 클 떄)

- 차원의 크기가 변하지 않음
- 기존의 정보를 상실할 가능성을 갖고있음


```python
Train_df = pd.concat([Train_X, Train_Y], axis = 1)
for col in Train_X.columns: # 보통은 범주 변수만 순회
    temp_dict = Train_df.groupby(col)['Class'].mean().to_dict()
    # col에 따른 Class의 평균을 나타내는 사전 (replace를 쓰기 위해, 사전으로 만듦)
    
    Train_df[col] = Train_df[col].replace(temp_dict) # 변수 치환    
    
    Test_X[col] = Test_X[col].astype(str).replace(temp_dict)
    # 테스트 데이터도 같이 치환해줘야 함 (나중에 활용하기 위해서는 저장도 필요)
```

    C:\ProgramData\Anaconda3\lib\site-packages\ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """



```python
Train_df.head()
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
      <th>Buying</th>
      <th>Maint</th>
      <th>Doors</th>
      <th>Persons</th>
      <th>Lug_boot</th>
      <th>Safety</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>821</th>
      <td>-0.85</td>
      <td>-0.810651</td>
      <td>-0.933014</td>
      <td>-0.875371</td>
      <td>-0.940887</td>
      <td>-0.897674</td>
      <td>1</td>
    </tr>
    <tr>
      <th>406</th>
      <td>-1.00</td>
      <td>-0.810651</td>
      <td>-0.934272</td>
      <td>-0.875371</td>
      <td>-0.940887</td>
      <td>-0.906103</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>299</th>
      <td>-1.00</td>
      <td>-1.000000</td>
      <td>-0.934272</td>
      <td>-0.875371</td>
      <td>-0.940887</td>
      <td>-0.897674</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>148</th>
      <td>-1.00</td>
      <td>-0.937107</td>
      <td>-0.938053</td>
      <td>-1.000000</td>
      <td>-0.928251</td>
      <td>-0.906103</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>840</th>
      <td>-0.85</td>
      <td>-0.810651</td>
      <td>-0.934272</td>
      <td>-0.875371</td>
      <td>-0.928251</td>
      <td>-1.000000</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>

</div>




```python
Train_X = Train_df.drop('Class', axis = 1)
Train_Y = Train_df['Class']
```


```python
# 치환한 뒤의 모델 테스트
model = KNN().fit(Train_X, Train_Y)
pred_Y = model.predict(Test_X)

f1_score(Test_Y, pred_Y)


# 라벨을 고려한 전처리이므로 더미화보다 좋은 결과가 나왔음 => 차원도 줄고 성능 상에 이점이 있으나, 
```




    0.0


