![다중공선성](C:\Users\훈\Desktop\다중공선성.PNG)



## 다중공선성과 변수 선택¶

다중공선성(multicollinearity)란 독립 변수의 일부가 다른 독립 변수의 조합으로 표현될 수 있는 경우이다. 독립 변수들이 서로 독립이 아니라 상호상관관계가 강한 경우에 발생한다. 이는 독립 변수의 공분산 행렬이 full rank 이어야 한다는 조건을 침해한다.

#### 다음 데이터는 미국의 거시경제지표를 나타낸 것이다.

- TOTEMP - Total Employment

- GNPDEFL - GNP deflator

- GNP - GNP

- UNEMP - Number of unemployed

- ARMED - Size of armed forces

- POP - Population

- YEAR - Year (1947 - 1962)




```python
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.datasets.longley import load_pandas

dfy = load_pandas().endog
dfX = load_pandas().exog
df = pd.concat([dfy, dfX], axis=1)
sns.pairplot(dfX)
# 스캐터 플롯에서 보듯이 독립변수간의 상관관계가 강하다.
```




    <seaborn.axisgrid.PairGrid at 0x18165823cc8>




![png](output_1_1.png)


### 상관관계는 상관계수 행렬로도 살펴볼 수 있다.


```python
dfX.corr() # 1에 가까울수록 상관관계가 높음
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
      <th>GNPDEFL</th>
      <th>GNP</th>
      <th>UNEMP</th>
      <th>ARMED</th>
      <th>POP</th>
      <th>YEAR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GNPDEFL</th>
      <td>1.000000</td>
      <td>0.991589</td>
      <td>0.620633</td>
      <td>0.464744</td>
      <td>0.979163</td>
      <td>0.991149</td>
    </tr>
    <tr>
      <th>GNP</th>
      <td>0.991589</td>
      <td>1.000000</td>
      <td>0.604261</td>
      <td>0.446437</td>
      <td>0.991090</td>
      <td>0.995273</td>
    </tr>
    <tr>
      <th>UNEMP</th>
      <td>0.620633</td>
      <td>0.604261</td>
      <td>1.000000</td>
      <td>-0.177421</td>
      <td>0.686552</td>
      <td>0.668257</td>
    </tr>
    <tr>
      <th>ARMED</th>
      <td>0.464744</td>
      <td>0.446437</td>
      <td>-0.177421</td>
      <td>1.000000</td>
      <td>0.364416</td>
      <td>0.417245</td>
    </tr>
    <tr>
      <th>POP</th>
      <td>0.979163</td>
      <td>0.991090</td>
      <td>0.686552</td>
      <td>0.364416</td>
      <td>1.000000</td>
      <td>0.993953</td>
    </tr>
    <tr>
      <th>YEAR</th>
      <td>0.991149</td>
      <td>0.995273</td>
      <td>0.668257</td>
      <td>0.417245</td>
      <td>0.993953</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
cmap = sns.light_palette("darkgray", as_cmap=True)
sns.heatmap(dfX.corr(), annot=True, cmap=cmap)
plt.show()
```


![png](output_4_0.png)


### 데이터셋을 나누고 모델을 fit 후 summary로 공분산 행렬의 조건수 확인


```python
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

def get_model1(seed):
    df_train, df_test = train_test_split(df, test_size=0.5, random_state=seed)
    model = sm.OLS.from_formula("TOTEMP ~ GNPDEFL + POP + GNP + YEAR + ARMED + UNEMP", data=df_train)
    return df_train, df_test, model.fit()


df_train, df_test, result1 = get_model1(3)
print(result1.summary()) # 다중공선성이 높다고 나옴!
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 TOTEMP   R-squared:                       1.000
    Model:                            OLS   Adj. R-squared:                  0.997
    Method:                 Least Squares   F-statistic:                     437.5
    Date:                Tue, 30 Mar 2021   Prob (F-statistic):             0.0366
    Time:                        13:51:36   Log-Likelihood:                -44.199
    No. Observations:                   8   AIC:                             102.4
    Df Residuals:                       1   BIC:                             103.0
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept  -1.235e+07   2.97e+06     -4.165      0.150      -5e+07    2.53e+07
    GNPDEFL      106.2620     75.709      1.404      0.394    -855.708    1068.232
    POP            2.2959      0.725      3.167      0.195      -6.915      11.506
    GNP           -0.3997      0.120     -3.339      0.185      -1.920       1.121
    YEAR        6300.6231   1498.900      4.203      0.149   -1.27e+04    2.53e+04
    ARMED         -0.2450      0.402     -0.609      0.652      -5.354       4.864
    UNEMP         -6.3311      1.324     -4.782      0.131     -23.153      10.491
    ==============================================================================
    Omnibus:                        0.258   Durbin-Watson:                   1.713
    Prob(Omnibus):                  0.879   Jarque-Bera (JB):                0.304
    Skew:                           0.300   Prob(JB):                        0.859
    Kurtosis:                       2.258   Cond. No.                     2.01e+10
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.01e+10. This might indicate that there are
    strong multicollinearity or other numerical problems.


    C:\ProgramData\Anaconda3\lib\site-packages\scipy\stats\stats.py:1604: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=8
      "anyway, n=%i" % int(n))


### 학습용 데이터와 검증용 데이터로 나누어 회귀분석 성능을 비교하면 과최적화가 발생하였음을 알 수 있음


```python
def calc_r2(df_test, result):
    target = df.loc[df_test.index].TOTEMP
    predict_test = result.predict(df_test)
    RSS = ((predict_test - target)**2).sum()
    TSS = ((target - target.mean())**2).sum()
    return 1 - RSS / TSS


test1 = []
for i in range(10):
    df_train, df_test, result = get_model1(i)
    test1.append(calc_r2(df_test, result))

test1
```




    [0.9815050656812714,
     0.9738497543138936,
     0.9879366369912537,
     0.7588861967909392,
     0.9807206089304153,
     0.8937889315213681,
     0.8798563810623438,
     0.9314665778977191,
     0.860852568224563,
     0.9677198735129439]



#### 독립변수가 서로 의존하게 되면 이렇게 과최적화(over-fitting) 문제가 발생하여 회귀 결과의 안정성을 해칠 가능성이 높아진다. 이를 방지하는 방법들은 다음과 같다.
- ##### Feature Selection: 중요 변수만 선택하는방법
 - 변수 선택법으로 의존적인 변수 삭제
 - Lasso
 - Stepwise
 - 정규화(regularized) 방법 사용

- ##### 변수를 줄이지 않고 활용하는 방법
 - AutoEncoder등 Feature Extraction 기법(딥러닝 기법)
 - PCA(principal component analysis) 방법으로 의존적인 성분 삭제
 - Ridge

##### 머신러닝 기법은 학습데이터 내에서 예측력을 높이기 위해 최대한 많은 변수를 활용


## VIF¶
다중 공선성을 없애는 가장 기본적인 방법은 다른 독립변수에 의존하는 변수를 없애는 것이다. 가장 의존적인 독립변수를 선택하는 방법으로는 VIF(Variance Inflation Factor)를 사용할 수 있다.


```python
# StatsModels에서는 variance_inflation_factor 으로 VIF를 계산한다.
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(
    dfX.values, i) for i in range(dfX.shape[1])]
vif["features"] = dfX.columns
vif # VIF가 10이 넘으면 다중공선성이 있는 변수라고 판단함(다른변수의 선형결합으로 x1을 설명하는 정도)
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
      <th>VIF Factor</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12425.514335</td>
      <td>GNPDEFL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10290.435437</td>
      <td>GNP</td>
    </tr>
    <tr>
      <th>2</th>
      <td>136.224354</td>
      <td>UNEMP</td>
    </tr>
    <tr>
      <th>3</th>
      <td>39.983386</td>
      <td>ARMED</td>
    </tr>
    <tr>
      <th>4</th>
      <td>101193.161993</td>
      <td>POP</td>
    </tr>
    <tr>
      <th>5</th>
      <td>84709.950443</td>
      <td>YEAR</td>
    </tr>
  </tbody>
</table>
</div>




```python
def get_model2(seed):
    df_train, df_test = train_test_split(df, test_size=0.5, random_state=seed)
    model = sm.OLS.from_formula("TOTEMP ~ scale(GNP) + scale(ARMED) + scale(UNEMP)", data=df_train)
    return df_train, df_test, model.fit()


df_train, df_test, result2 = get_model2(3)
print(result2.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 TOTEMP   R-squared:                       0.989
    Model:                            OLS   Adj. R-squared:                  0.981
    Method:                 Least Squares   F-statistic:                     118.6
    Date:                Tue, 30 Mar 2021   Prob (F-statistic):           0.000231
    Time:                        14:09:13   Log-Likelihood:                -57.695
    No. Observations:                   8   AIC:                             123.4
    Df Residuals:                       4   BIC:                             123.7
    Df Model:                           3                                         
    Covariance Type:            nonrobust                                         
    ================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
    --------------------------------------------------------------------------------
    Intercept     6.538e+04    163.988    398.686      0.000    6.49e+04    6.58e+04
    scale(GNP)    4338.7051    406.683     10.669      0.000    3209.571    5467.839
    scale(ARMED)  -812.1407    315.538     -2.574      0.062   -1688.215      63.933
    scale(UNEMP) -1373.0426    349.316     -3.931      0.017   -2342.898    -403.187
    ==============================================================================
    Omnibus:                        0.628   Durbin-Watson:                   2.032
    Prob(Omnibus):                  0.731   Jarque-Bera (JB):                0.565
    Skew:                           0.390   Prob(JB):                        0.754
    Kurtosis:                       1.958   Cond. No.                         4.77
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


    C:\ProgramData\Anaconda3\lib\site-packages\scipy\stats\stats.py:1604: UserWarning: kurtosistest only valid for n>=20 ... continuing anyway, n=8
      "anyway, n=%i" % int(n))


### 다중공선성을 제거한 경우에는 학습 성능과 검증 성능간의 차이가 줄어들었음을 확인할 수 있다. 즉, 과최적화가 발생하지 않는다.


```python
plt.subplot(121)
plt.plot(test1, 'ro', label="검증 성능")
plt.hlines(result1.rsquared, 0, 9, label="학습 성능")
plt.legend()
plt.xlabel("시드값")
plt.ylabel("성능(결정계수)")
plt.title("다중공선성 제거 전")
plt.ylim(0.5, 1.2)

plt.subplot(122)
plt.plot(test2, 'ro', label="검증 성능")
plt.hlines(result2.rsquared, 0, 9, label="학습 성능")
plt.legend()
plt.xlabel("시드값")
plt.ylabel("성능(결정계수)")
plt.title("다중공선성 제거 후")
plt.ylim(0.5, 1.2)

plt.suptitle("다중공선성 제거 전과 제거 후의 성능 비교", y=1.04)
plt.tight_layout()
plt.show()
```

![다중2](C:\Users\훈\Desktop\다중2.PNG)


