## 기본 패키지
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm

# 데이터 분류
from sklearn.model_selection import train_test_split

# 모델 평가
from sklearn.metrics import classification_report
from sklearn import metrics

data = pd.read_csv('winequality-red.csv')

# 데이터셋의 'quality' 는 2~8
# 'quality' 값의  6 점을 기준으로 좋은 와인과 나쁜 와인을 구분하겠다고 선언
# pandas.cut(데이터, 구간 개수, 구분할 데이터가 들어갈 레이블명) 함수를 활용하여
# 수치가 다양한 label 값을 good 과 bad 2가지로 분류 // 속성 bins : 나누고자하는 구간 개수
standard = (2, 6, 8)
groupNames = ['bad', 'good']
data['quality'] = pd.cut(data['quality'], bins=standard, labels=groupNames)

# 데이터와 레이블을 구분
train = data.drop('quality', axis=1)
label = data['quality']

# 훈련, 테스트 할 데이터 / 레이블 분류
train_data, test_data, train_label, test_label = \
    train_test_split(train, label, test_size=0.2, random_state=42)


# 학습 및 예측
clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)


# 정확도 출력
score = metrics.accuracy_score(test_label, pre)
report = classification_report(test_label, pre)
print(f"정확도 : {score*100}%")

print(f"결과 {report}")

# 시각화
# data = pd.read_csv('./winequality-red.csv')

fig, ax = plt.subplots(ncols=3, nrows=2)
sns.regplot(x='pH', y='fixed acidity', data=data, ax=ax[0][0])  # x : 산성도 y : 고정 산도
sns.regplot(x='pH', y='alcohol', data=data, ax=ax[0][1])        # x : 산성도 y : 알콜도수
sns.regplot(x='pH', y='density', data=data, ax=ax[0][2])        # x : 품질 y : 밀도
sns.barplot(x='quality', y='alcohol', data=data, ax=ax[1][0])   # x : 품질 y : 알콜도수
sns.barplot(x='quality', y='total sulfur dioxide', data=data, ax=ax[1][1])  # x : 품질 y : 이산화황
sns.countplot(x='quality', data=data, ax=ax[1][2]) # 품질에따른 수량

plt.show()


