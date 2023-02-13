# scikit-learn Note

安装：

`pip install -U scikit-learn`

```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
X = [[1, 2, 3],
    [11, 12, 13]]  # 2 samples, 3 features
y = [0, 1]  # classes of each sample
clf.fit(X, y)
clf.predict(X)  # predict classes of the training data
clf.predict([[4, 5, 6], [14, 15, 16]])  # predict classes of new data
```

transformers and pre-processors:

这些模块可以用 pipeline 预处理数据。和 mmdet 挺像的，不知道 sklearn 和 mmdet 他俩谁抄的谁。

```python
from sklearn.preprocessing import StandardScaler
X = [[0, 15],
    [1, -10]]
# scale data according to computed scaling values
StandardScaler().fit(X).transform(X)
```

输出：

```
array([-1., 1.],
    [1., -1.])
```

注：

1. pre-processors, transformers 和 extimators 都继承自`BaseEstimator`。感觉这个设计挺智障的。

pipelines:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticREgression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# create a pipeline object
pipe = make_pipeline(
    StandardScaler(),
    LOgisticRergression()
)

# Load the iris
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# fit the whole pipeline
pipe.fit(X_train, y_train)
accuracy_score(pipe.predict(X_test), y_test)
```

Model evaluation:

```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

X, y = make_regression(n_samples=1000, random_state=0)
lr = LinearRegression()

result = cross_validate(lr, X, y)  # defaults to 5-fold CV
result['test_score']  # r_squared score is high because dataset is easy
```

Automatic parameter searches

```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizeSearchCV, train_test_split
from scipy.stats import randint

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# define the parameter space that will be searched over
param_distributions = {'n_estimators': randint(1, 5), 'max_depth': randint(5, 10)}

# now create a searchCV object and fit it to the data
search = RandomizedSearchCV(estimator=RandomForestRegression(random_state=0), n_iter=5, param_distributions=param_distributions, random_state=0)
search.fit(X_train, y_train)
search.best_params_

# the search object now acts like a normal random forest estimator
# with max_depth=9 and n_estimators=4
search.score(X_test, y_test)
```