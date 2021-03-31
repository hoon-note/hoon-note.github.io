---
layout: single
title: 'Python Over&Under_Sampling'
---

#### 클래스 불균형(분류 문제에서 발생)
- 클래스 변수가 하나의 값에 치우친 데이터로 학습한 분류 모델이 치우친 클래스에 대해 편향되는 문제
- 이런 모델은 대부분 샘플을 치우친 클래스 값으로만 분류함(ex:암환자 판별 문제)
- 실제로 암 환자인 경우 9999 아닌 경우가 1 이라면 정확도는 높지만 재현율이 매우 낮음

#####  ***클래스 불균형 문제가 있는 모델은 정확도가 높고 재현율이 매우 낮은 경향이 있음***


![image-center](/assets/images/o1.PNG){: .align-center}
#### 용어 정의
![image-center](/assets/images/o4.PNG){: .align-center}

  - 다수 클래스 :대부분 샘플이 속한 클래스  [부정 클래스] (예:정상인)
  - 소수 클래스 :대부분 샘플에 속하지 않은 클래스 [긍정 클래스] (예:암환자)
  - 위양성 비용(TP) :부정 클래스 샘플을 긍정 클래스 샘플로 분류해서 발생하는 비용
  - 위음성 비용(TN) :긍정 클래스 샘플을 부정 클래스 샘플로 분류해서 발생하는 비용
  - 보통은 위음성 비용이 위양성 비용보다 훨씬 큼(정산인 -> 암환자 vs 암환자 ->정상인) 돈,시간 vs 생명
  - 절대 부족 :소수 클래스에 속한 샘플 개수가 절대적으로 부족한 상황

#### 탐색 방법
 - 클래스 불균형 비율이 9 이상이면 편향된 모델이 학습될 가능성이 있음

 - 클래스 불균형 비율 =  다수 클래스에 속한 샘플 수 / 소수 클래스에 속한 샘플 수 

 - 하지만,클래스 불균형 비율이 높다고 해서 반드시 편향된 모델을 학습하는 것은 아님

 - K-최근접 이웃은 이웃의 클래스 정보를 바탕으로 분류하기에 클래스 불균형에 매우 민감하므로 불균형 문제를 진단하는데 적절함


![image-center](/assets/images/o2.PNG){: .align-center}
 - k값이 크면 클수록 더욱 민감하므로,보통 5~11 정도의 k를 설정하여 문제를 진단함 (KNN의 재현율 확인)

#### 문제 해결의 기본 (재현율을 올리는것)
 - 소수 클래스에 대한 결정 공간을 넓히는것 
![image-center](/assets/images/o3.PNG){: .align-center}


   
![image-center](/assets/images/o5.PNG){: .align-center}


#### 오버샘플링 (평가 데이터에 대해서는 절대로 재샘플링을 하면 안 됨)

- 소수 클래스 샘플 생성(원본 데이터가 작을 때 사용(계산량문제가 있음) 가짜데이터를 만듬
- 결정 경계에 가까운 다수 클래스 샘플을 제거하고,결정 경계에 가까운 소수 클래스 샘플을 생성해야 함


```python
import pandas as pd
import os
import numpy as np

```


```python
df = pd.read_csv("Secom.csv")
df.shape
```




    (1567, 591)




```python
# 특징과 라벨 분리
X = df.drop('Y', axis = 1)
Y = df['Y']
```


```python
# 학습 데이터와 평가 데이터 분할
from sklearn.model_selection import train_test_split
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y)
```


```python
# 특징이 매우 많음을 확인
Train_X.shape
```




    (1175, 590)




```python
# 클래스 불균형 확인 => 언더샘플링을 적용하기에는 부적절 
Train_Y.value_counts()
```


    -1    1095
     1      80
    Name: Y, dtype: int64


```python
# 클래스 불균형 비율 계산 (9이상이면 높음)
Train_Y.value_counts().iloc[0] / Train_Y.value_counts().iloc[-1]
```


    13.6875


```python
# kNN을 사용한 클래스 불균형 테스트
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import *

kNN_model = KNN(n_neighbors = 11).fit(Train_X, Train_Y)
pred_Y = kNN_model.predict(Test_X)
print(recall_score(Test_Y, pred_Y)) # 재현율
print(accuracy_score(Test_Y, pred_Y)) # 정확도

# 재현율이 0%로 불균형이 심각한 수준이라 보임
```

    0.0
    0.9387755102040817
![image-center](/assets/images/smote.PNG){: .align-center}


```python
from imblearn.over_sampling import SMOTE
# SMOTE 인스턴스 생성
oversampling_instance = SMOTE(k_neighbors = 3) #고려하는 이웃 수 설정

# 오버샘플링 적용
o_Train_X, o_Train_Y = oversampling_instance.fit_sample(Train_X, Train_Y) # ndarray #TEST_X는 불가

# ndarray 형태가 되므로 다시 DataFrame과 Series로 변환 (남은 전처리가 없다면 하지 않아도 무방)
o_Train_X = pd.DataFrame(o_Train_X, columns = X.columns)
o_Train_Y = pd.Series(o_Train_Y)
```


```python
# 비율이 1:1이 됨을 확인 (가짜데이터가 만들어짐)
o_Train_Y.value_counts()
```




     1    1095
    -1    1095
    Name: Y, dtype: int64




```python
# 같은 모델로 다시 평가: 정확도는 감소했으나, 재현율이 크게 오름을 확인
kNN_model = KNN(n_neighbors = 11).fit(o_Train_X, o_Train_Y)
pred_Y = kNN_model.predict(Test_X)
print(recall_score(Test_Y, pred_Y))
print(accuracy_score(Test_Y, pred_Y))
```

    0.4583333333333333
    0.5204081632653061



```python
# 정확도를 올리기위한 오버샘플링
from imblearn.over_sampling import SMOTE
# SMOTE 인스턴스 생성
oversampling_instance = SMOTE(k_neighbors = 3, # 2로나눠서 INT형이 아닌것을 INT로 만듬
               sampling_strategy = {1:int(Train_Y.value_counts().iloc[0] / 2),#절반으로가짜데이터만듬
                              -1:Train_Y.value_counts().iloc[0]}) #부정클래스

# 오버샘플링 적용
o_Train_X, o_Train_Y = oversampling_instance.fit_sample(Train_X, Train_Y)

# ndarray 형태가 되므로 다시 DataFrame과 Series로 변환 (남은 전처리가 없다면 하지 않아도 무방)
o_Train_X = pd.DataFrame(o_Train_X, columns = X.columns)
o_Train_Y = pd.Series(o_Train_Y)
```


```python
kNN_model = KNN(n_neighbors = 11).fit(o_Train_X, o_Train_Y)
pred_Y = kNN_model.predict(Test_X)
print(recall_score(Test_Y, pred_Y))
print(accuracy_score(Test_Y, pred_Y))
# 가짜데이터를 절반으로 만든 후의 값은 재현율이 떨어지고 정확도가 올라감
# 두 개의 시나리오중 적절하게 판단하여 선택
```

    0.375
    0.6785714285714286



#### 언더샘플링

- 다수 클래스 샘플 삭제(원본 데이터가 클 때)


```python
import pandas as pd
import os
import numpy as np

```


```python
df = pd.read_csv("page-blocks0.csv")
df.shape
```




    (5472, 11)




```python
# 특징과 라벨 분리
X = df.drop('Class', axis = 1)
Y = df['Class']
```


```python
# 학습 데이터와 평가 데이터 분할
from sklearn.model_selection import train_test_split
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X, Y)
```


```python
# 클래스 불균형 확인
Train_Y.value_counts()
```


    negative    3698
    positive     406
    Name: Class, dtype: int64


```python
Train_Y.replace({"negative":-1, "positive":1}, inplace = True)
Test_Y.replace({"negative":-1, "positive":1}, inplace = True)
```


```python
# 클래스 불균형 비율 계산
Train_Y.value_counts().iloc[0] / Train_Y.value_counts().iloc[-1]
```


    9.108374384236454


```python
# kNN을 사용한 클래스 불균형 테스트
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import *

kNN_model = KNN(n_neighbors = 11).fit(Train_X, Train_Y)
pred_Y = kNN_model.predict(Test_X)
print(recall_score(Test_Y, pred_Y))
print(accuracy_score(Test_Y, pred_Y))

# 재현율이 64%로 불균형이 심각한 수준은 아니라고 보임
```

    0.6470588235294118
    0.9532163742690059
![image-center](/assets/images/Near.PNG){: .align-center}

```python
from imblearn.under_sampling import NearMiss
NM_model = NearMiss(version = 2) # version = 2: 모든 소수 클래스 샘플까지의 평균 거리를 활용

# NearMiss 적용
u_Train_X, u_Train_Y = NM_model.fit_sample(Train_X, Train_Y)
u_Train_X = pd.DataFrame(u_Train_X, columns = X.columns)
u_Train_Y = pd.Series(u_Train_Y)
```


```python
# 1:1 비율
u_Train_Y.value_counts()
```




     1    406
    -1    406
    Name: Class, dtype: int64




```python
# kNN 재적용을 통한 성능 변화 확인
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import *

kNN_model = KNN(n_neighbors = 11).fit(u_Train_X, u_Train_Y)
pred_Y = kNN_model.predict(Test_X)
print(recall_score(Test_Y, pred_Y))
print(accuracy_score(Test_Y, pred_Y))

# 재현율은 크게 올랐으나, 정확도가 크게 떨어짐 => 적당한 비율에 맞게 설정해야 함
# 소수클래스의 결정공간이 너무 커져서 다수 클래스를 판단하지 못함
```

    0.9607843137254902
    0.22514619883040934



```python
from imblearn.under_sampling import NearMiss
NM_model = NearMiss(version = 2, sampling_strategy = {1:u_Train_Y.value_counts().iloc[-1],
                                                      -1:u_Train_Y.value_counts().iloc[-1] * 5})
                                                     # 5:1 정도의 비율로 언더샘플링 재수행
                                                     # 다수클래스 5: 소수클래스 1 
u_Train_X, u_Train_Y = NM_model.fit_sample(Train_X, Train_Y)
u_Train_X = pd.DataFrame(u_Train_X, columns = X.columns)
u_Train_Y = pd.Series(u_Train_Y)
```


```python
u_Train_Y.value_counts()
```




    -1    2030
     1     406
    Name: Class, dtype: int64




```python
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import *
kNN_model = KNN(n_neighbors = 11).fit(u_Train_X, u_Train_Y)
pred_Y = kNN_model.predict(Test_X)
print(recall_score(Test_Y, pred_Y))
print(accuracy_score(Test_Y, pred_Y))
```

    0.8627450980392157
    0.6630116959064327


