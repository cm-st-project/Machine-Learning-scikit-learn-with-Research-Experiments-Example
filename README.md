
# Machine-Learning-scikit-learn-with-Research-Experiments-Example
## Example 1: Level of performace of the Volunteer Opportunity 
The specific objective in this example is to predict the performance level required for individuals to participate in the Volunteer Opportunity.
### Dummy Data
1. Identify the problem domain for machine learning (ML) application.
   - To identify the problem domain for an ML application focused onperformance in the Volunteer Opportunity, follow these steps:
      - Define the goals and objectives of the ML application in relation to performance in volunteer opportunities.
      - Gather relevant data related to volunteer performance, such as volunteer profiles, activities, and outcomes.
      - Determine relevant input features: age, gender, average, eventType, time, race and major/subject.
2. Formulate the dataset structure, including input features and the target result to predict.
    - For the first dataset,
      - input: gender (female/male), average(1-10), age (12-19), event type(1-6), science(yes/no), computer Science(yes/no), writing(yes/no), public Speaking, math(yes/no), engineering(yes/no)
      - output: performace (1-10)
    - For the second dataset
      - input: age (12-19), event type(1-6), science(yes/no), computer Science(yes/no), writing(yes/no), public Speaking, math(yes/no), engineering(yes/no)
      - output: performace (1-10)
3. Generate dummy data to simulate real-world scenarios.
```python
input_data = []
output_data = []

for i in range(total_dataset):
  age = random.randint(12, 19)
  # gender = random.randint(1, 2)
  average = random.randint(1, 10)
  eventType = random.randint(1, 6)
  hrsReq = random.randint(1, 15)
  # race = random.randint(1, 5)
  science = random.randint(1, 2)
  compSci = random.randint(1, 2)
  writing = random.randint(1, 2)
  publicSpeaking = random.randint(1, 2)
  math = random.randint(1, 2)
  engineering = random.randint(1, 2)

  # First dataset
  # input_data.append([age, gender, average, eventType, hrsReq, race, science, compSci, writing, publicSpeaking, math, engineering])
  # Second Dataset
  input_data.append([age, eventType, science, compSci, writing, publicSpeaking, math, engineering])

  # Define the minimum performance value for the specific type of volunteer which depends on the age, major/subject, event type and time.
  min_per = 1
  min_per = average + random.randint(-2, 2)
  if age > 15:
    min_per += age - 15
  if science + compSci + writing + math + publicSpeaking + engineering >= 10:
    min_per += 3
  
  if eventType == 1 and science == 2:
    min_per += 1

  if eventType == 2 and compSci == 2:
    min_per += 1

  if eventType == 3 and writing == 2:
    min_per += 1
  
  if eventType == 4 and publicSpeaking == 2:
    min_per += 1
  
  if eventType == 5 and math == 2:
    min_per += 1
  
  if eventType == 6 and engineering == 2:
    min_per += 1

  if min_per > 10:
    min_per = 9
  
  '''
  if hrsReq > 10:
    min_per -= random.randint(0, 1)

  if hrsReq > 5:
    min_per -= random.randint(0, 1)
'''

  performance = random.randint(min_per, 10)
  if performance == 0:
    performance = 1

  # print(age, gender, average, eventType, hrsReq, race, science, compSci, writing, publicSpeaking, math, engineering, " => ", performance)
  # print(age, eventType, science, compSci, writing, publicSpeaking, math, engineering, " => ", performance)

  output_data.append(performance)
```
#### Dummy Data Generation Tips

- Complete randomness is not suitable for generating data in ML because the model cannot learn from it effectively.
- It is important to control the random generation process based on the expected rules. For example, if you believe that feature X will positively impact the result Z, you should generate a dataset that adheres to that rule.

### Train & Prediction
1. Train the ML model using the prepared dataset.
2. Perform predictions using the trained model on new data.
``` python
# Train the model
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=0)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)
print(f"{model.score(X_test, y_test)}")

# Do Prediction
# [age, eventType, science, compSci, writing, publicSpeaking, math, engineering]
test_data = [15,4,1,1,2,2,1,2]
pred = model.predict([test_data])
print(pred)

```

### Experiments
1. Experiment with 2-3 other ML algorithms and compare their accuracy using cross-validation techniques.
```python
   
  X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=0)
  
  model = linear_model.LinearRegression()
  model.fit(X_train, y_train)
  print('Res:', model.score(X_test, y_test))
  
  model3 = svm.SVC()
  model3.fit(X_train, y_train)
  print(f"{model3.score(X_test, y_test)}")
  
  model4 = RandomForestClassifier(n_estimators=30, max_depth=30)
  model4.fit(X_train, y_train)
  print(f"{model4.score(X_test, y_test)}")

```
2. Conduct experiments using the chosen ML algorithm but with different parameters or configurations, comparing accuracy through cross-validation.
```python
  #kernel parameters selects the type of hyperplane used to separate the data. Using 'linear' will use a linear hyperplane (a line in the case of 2D data).        #'rbf' and 'poly' uses a non linear hyper-plane
  # kernels = ['linear', 'rbf', 'poly']
  for kernel in kernels:
    svc = svm.SVC(kernel=kernel)
    svc.fit(X_train, y_train)
    print(f"{kernel}: {svc.score(X_test, y_test)}")
  
  # gamma is a parameter for non linear hyperplanes. The higher the gamma value it tries to exactly fit the training data set
  gammas = [0.1, 1, 10, 100]
  for gamma in gammas:
    svc = svm.SVC(kernel='rbf', gamma=gamma)
    svc.fit(X_train, y_train)
    print(f"{gamma}: {svc.score(X_test, y_test)}")
  
  #C is the penalty parameter of the error term. It controls the trade off between smooth decision boundary and classifying
  #the training points correctly.
  cs = [0.1, 1, 10, 100, 1000]
  for c in cs:
    svc = svm.SVC(kernel='rbf', C=c)
    svc.fit(X_train, y_train)
    print(f"{c}: {svc.score(X_test, y_test)}")
    
  #degree is a parameter used when kernel is set to ‘poly’. It’s basically the degree of the polynomial used 
  #to find the hyperplane to split the data.
  degrees = [0, 1, 2, 3, 4, 5, 6]
  for degree in degrees:
    svc = svm.SVC(kernel='poly', degree=degree)
    svc.fit(X_train, y_train)
    print(f"{degree}: {svc.score(X_test, y_test)}")

```
3. Perform experiments with different datasets, varying the features/columns, and assess accuracy using cross-validation for comparison.

```
total_dataset = 10000
input_data1 = []
input_data2 = []
output_data1 = []
output_data2 = []

for i in range(total_dataset):
  age = random.randint(12, 19)
  # gender = random.randint(1, 2)
  average = random.randint(1, 10)
  eventType = random.randint(1, 6)
  hrsReq = random.randint(1, 15)
  # race = random.randint(1, 5)
  science = random.randint(1, 2)
  compSci = random.randint(1, 2)
  writing = random.randint(1, 2)
  publicSpeaking = random.randint(1, 2)
  math = random.randint(1, 2)
  engineering = random.randint(1, 2)
  # finish hex
  input_data1.append([age, gender, average, eventType, hrsReq, race, science, compSci, writing, publicSpeaking, math, engineering])
  input_data2.append([age, eventType, science, compSci, writing, publicSpeaking, math, engineering])

  min_per = 1
  min_per = average + random.randint(-2, 2)
  if age > 15:
    min_per += age - 15
  if science + compSci + writing + math + publicSpeaking + engineering >= 10:
    min_per += 3
  
  if eventType == 1 and science == 2:
    min_per += 1

  if eventType == 2 and compSci == 2:
    min_per += 1

  if eventType == 3 and writing == 2:
    min_per += 1
  
  if eventType == 4 and publicSpeaking == 2:
    min_per += 1
  
  if eventType == 5 and math == 2:
    min_per += 1
  
  if eventType == 6 and engineering == 2:
    min_per += 1

  if min_per > 10:
    min_per = 9

  performance = random.randint(min_per, 10)
  if performance == 0:
    performance = 1

  output_data2.append(performance)
 
  if hrsReq > 10:
    min_per -= random.randint(0, 1)

  if hrsReq > 5:
    min_per -= random.randint(0, 1)


  performance = random.randint(min_per, 10)
  if performance == 0:
    performance = 1

  output_data1.append(performance)


X_train, X_test, y_train, y_test = train_test_split(input_data1, output_data1, test_size=0.2, random_state=0)
model1 = svm.SVC()
model1.fit(X_train, y_train)
print(f"{model.score(X_test, y_test)}")

X_train, X_test, y_train, y_test = train_test_split(input_data2, output_data2, test_size=0.2, random_state=0)
model2 = svm.SVC()
model2.fit(X_train, y_train)
print(f"{model.score(X_test, y_test)}")
```
### Source Code
```python
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
import random

# 1. Focus on classification
# 2. Identify 5 different popular classication models
#  - Experiment 1: Acc using different models
#  - Experiment 2: Acc using different feature set
#  - Experiment 3: Acc using different model parameters
#  - Experiment 4: Acc using different training data sizes


total_dataset = 10000
input_data = []
output_data = []

for i in range(total_dataset):
  age = random.randint(12, 19)
  # gender = random.randint(1, 2)
  average = random.randint(1, 10)
  eventType = random.randint(1, 6)
  hrsReq = random.randint(1, 15)
  # race = random.randint(1, 5)
  science = random.randint(1, 2)
  compSci = random.randint(1, 2)
  writing = random.randint(1, 2)
  publicSpeaking = random.randint(1, 2)
  math = random.randint(1, 2)
  engineering = random.randint(1, 2)
  # finish hex
  # input_data.append([age, gender, average, eventType, hrsReq, race, science, compSci, writing, publicSpeaking, math, engineering])
  input_data.append([age, eventType, science, compSci, writing, publicSpeaking, math, engineering])

  min_per = 1
  min_per = average + random.randint(-2, 2)
  if age > 15:
    min_per += age - 15
  if science + compSci + writing + math + publicSpeaking + engineering >= 10:
    min_per += 3
  
  if eventType == 1 and science == 2:
    min_per += 1

  if eventType == 2 and compSci == 2:
    min_per += 1

  if eventType == 3 and writing == 2:
    min_per += 1
  
  if eventType == 4 and publicSpeaking == 2:
    min_per += 1
  
  if eventType == 5 and math == 2:
    min_per += 1
  
  if eventType == 6 and engineering == 2:
    min_per += 1

  if min_per > 10:
    min_per = 9
    '''
  if hrsReq > 10:
    min_per -= random.randint(0, 1)

  if hrsReq > 5:
    min_per -= random.randint(0, 1)
'''

  performance = random.randint(min_per, 10)
  if performance == 0:
    performance = 1

  # print(age, gender, average, eventType, hrsReq, race, science, compSci, writing, publicSpeaking, math, engineering, " => ", performance)
  # print(age, eventType, science, compSci, writing, publicSpeaking, math, engineering, " => ", performance)

  output_data.append(performance)


X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=0)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)
print(f"{model.score(X_test, y_test)}")

model2 = make_pipeline(PolynomialFeatures(2), Ridge())
model2.fit(X_train, y_train)
print(f"{model2.score(X_test, y_test)}")


model3 = svm.SVC()
model3.fit(X_train, y_train)
print(f"")


model4 = RandomForestClassifier(n_estimators=30, max_depth=30)
model4.fit(X_train, y_train)
print(f"{model4.score(X_test, y_test)}")


model5 = KNeighborsClassifier(n_neighbors=3)
model5.fit(X_train, y_train)
print(f"{model5.score(X_test, y_test)}")

model6 = GaussianNB()
model6.fit(X_train, y_train)
print(f"{model6.score(X_test, y_test)}")


model7 = BaggingClassifier(RandomForestClassifier(), max_samples=0.5, max_features=0.5)
model7.fit(X_train, y_train)
print(f"{model7.score(X_test, y_test)}")
```
## Example 2: Prediction of Covid Risk based on Location and Activity
For this particular example, the goal is to predict the risk of Covid based on location and activity.

```python
# https://scikit-learn.org/stable/
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import random 

# Experiment 1
input_data = []
input_data2 = []
output_data = []

# randomly generate dummy test data
total_records = 5000
for i in range(total_records):
  lat = random.uniform(30, 34)
  lon = random.uniform(-120, -117)
  locs = random.randint(3, 20)
  events = random.randint(0, 1)
  input_data.append([lat, lon, locs, events])
  input_data2.append([locs, events])

  # risk 1-10
  min_risk = 1
  if events == 1:
    min_risk = 5
  if locs > 10:
    min_risk = min_risk + 3
  risk = random.randint(min_risk, 10)
  # if lat == 30:
  #   risk = 7
  # if lat == 31:
  #   risk = 8
  # if lat == 32:
  #   risk = 9
  # if lat >= 33:
  #   risk = 10
  output_data.append(risk)
  # print(lat, lon, locs, events, ' => ', risk)
  

X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.3, random_state=0)

X_train2, X_test2, y_train2, y_test2 = train_test_split(input_data2, output_data, test_size=0.3, random_state=0)

# input_data = [
#   #[31, -118, 0, 11],
#   #[32, -117, 1, 15],
#   [33, -116.75, 0, 18],
#   [31.35, -115.89, 1, 6],
#   [32, -118, 1, 16],
#   [31, -117, 0, 10]
# ]

# output_data = [
#   #5,
#   #9,
#   9.5,
#   9,
#   7,
#   5
# ]

# Experiment 1-1
print("Experiment 1-1")
model = linear_model.LinearRegression()
model.fit(X_train, y_train) # learn, train, fit
print('Res:', model.score(X_test, y_test))

model2 = make_pipeline(PolynomialFeatures(2), Ridge())
model2.fit(X_train, y_train)
print('Res:', model2.score(X_test, y_test))

model3 = LogisticRegression(random_state=0)
model3.fit(X_train, y_train)
print('Res:', model3.score(X_test, y_test))

model4 = RandomForestRegressor(max_depth=2, random_state=0)
model4.fit(X_train, y_train)
print('Res:', model4.score(X_test, y_test))



# Experiment 1-2
print("Experiment 1-2")
model2 = make_pipeline(PolynomialFeatures(2), Ridge())
model2.fit(X_train, y_train)
# print(model2.predict([ [31, -118, 0, 11] ]))
print('Res:', model2.score(X_test, y_test))

model2 = make_pipeline(PolynomialFeatures(3), Ridge())
model2.fit(X_train, y_train)
# print(model2.predict([ [31, -118, 0, 11] ]))
print('Res:', model2.score(X_test, y_test))

model2 = make_pipeline(PolynomialFeatures(4), Ridge())
model2.fit(X_train, y_train)
# print(model2.predict([ [31, -118, 0, 11] ]))
print('Res:', model2.score(X_test, y_test))

model2 = make_pipeline(PolynomialFeatures(5), Ridge())
model2.fit(X_train, y_train)
# print(model2.predict([ [31, -118, 0, 11] ]))
print('Res:', model2.score(X_test, y_test))

# Experiment 1-3
print("Experiment 1-3")
model2 = make_pipeline(PolynomialFeatures(2), Ridge())
model2.fit(X_train2, y_train2)
# print(model2.predict([ [31, -118, 0, 11] ]))
print('Res:', model2.score(X_test2, y_test2))

model2 = make_pipeline(PolynomialFeatures(3), Ridge())
model2.fit(X_train2, y_train2)
# print(model2.predict([ [31, -118, 0, 11] ]))
print('Res:', model2.score(X_test2, y_test2))

model2 = make_pipeline(PolynomialFeatures(4), Ridge())
model2.fit(X_train2, y_train2)
# print(model2.predict([ [31, -118, 0, 11] ]))
print('Res:', model2.score(X_test2, y_test2))

model2 = make_pipeline(PolynomialFeatures(5), Ridge())
model2.fit(X_train2, y_train2)
# print(model2.predict([ [31, -118, 0, 11] ]))
print('Res:', model2.score(X_test2, y_test2))


# Experiment 2
input_data2 = []
output_data2 = []
input_data3 = []

total_records = 5000
for i in range(total_records):
  lat = random.uniform(30, 34)
  lon = random.uniform(-120, -117)
  type = random.randint(1, 3)
  locs = random.randint(3, 40)
  visits = random.randint(10, 50)
  hours_spent = random.uniform(0.5, 5)
  input_data2.append([type, locs, visits, hours_spent])
  input_data3.append([type, locs, hours_spent])

  # risk 1-10
  min_risk = 0
  min_risk += type
  if hours_spent >= 2:
    min_risk += 1
  if locs > 10:
    min_risk = min_risk + 1
  risk = random.randint(min_risk, 5)
  output_data2.append(risk)
  
# print(input_data2)
# print(output_data2)


# model = linear_model.LinearRegression()
# model.fit(input_data2, output_data2) # learn, train, fit
# # print(model.predict([ [31, -118.32, 16, 40, 3] ] ))

# model2 = make_pipeline(PolynomialFeatures(2), Ridge())
# model2.fit(input_data2, output_data2)
# # print(model2.predict([ [31, -118.32, 16, 40, 3] ]))

X_train2, X_test2, y_train2, y_test2 = train_test_split(input_data2, output_data2, test_size=0.3, random_state=0)

X_train3, X_test3, y_train3, y_test3 = train_test_split(input_data3, output_data2, test_size=0.3, random_state=0)

# Experiment 2-1
print("Experiment 2-1")
model = linear_model.LinearRegression()
model.fit(X_train2, y_train2) # learn, train, fit
print('Res:', model.score(X_test2, y_test2))

model2 = make_pipeline(PolynomialFeatures(2), Ridge())
model2.fit(X_train2, y_train2)
print('Res:', model2.score(X_test2, y_test2))

model3 = LogisticRegression(random_state=0)
model3.fit(X_train2, y_train2)
print('Res:', model3.score(X_test2, y_test2))

model4 = RandomForestRegressor(max_depth=2, random_state=0)
model4.fit(X_train2, y_train2)
print('Res:', model4.score(X_test2, y_test2))



# Experiment 2-2
print("Experiment 2-2")
model2 = make_pipeline(PolynomialFeatures(2), Ridge())
model2.fit(X_train2, y_train2)
# print(model2.predict([ [31, -118, 0, 11] ]))
print('Res:', model2.score(X_test2, y_test2))

model2 = make_pipeline(PolynomialFeatures(3), Ridge())
model2.fit(X_train2, y_train2)
# print(model2.predict([ [31, -118, 0, 11] ]))
print('Res:', model2.score(X_test2, y_test2))

model2 = make_pipeline(PolynomialFeatures(4), Ridge())
model2.fit(X_train2, y_train2)
# print(model2.predict([ [31, -118, 0, 11] ]))
print('Res:', model2.score(X_test2, y_test2))

model2 = make_pipeline(PolynomialFeatures(5), Ridge())
model2.fit(X_train2, y_train2)
# print(model2.predict([ [31, -118, 0, 11] ]))
print('Res:', model2.score(X_test2, y_test2))

# Experiment 2-3
print("Experiment 2-3")
model2 = make_pipeline(PolynomialFeatures(2), Ridge())
model2.fit(X_train3, y_train3)
# print(model2.predict([ [31, -118, 0, 11] ]))
print('Res:', model2.score(X_test3, y_test3))

model2 = make_pipeline(PolynomialFeatures(3), Ridge())
model2.fit(X_train3, y_train3)
# print(model2.predict([ [31, -118, 0, 11] ]))
print('Res:', model2.score(X_test3, y_test3))

model2 = make_pipeline(PolynomialFeatures(4), Ridge())
model2.fit(X_train3, y_train3)
# print(model2.predict([ [31, -118, 0, 11] ]))
print('Res:', model2.score(X_test3, y_test3))

model2 = make_pipeline(PolynomialFeatures(5), Ridge())
model2.fit(X_train3, y_train3)
# print(model2.predict([ [31, -118, 0, 11] ]))
print('Res:', model2.score(X_test3, y_test3))


```


## Example 3: Prediction of Vocabulary Learning Pattern

```python
from sklearn import svm
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import dataGeneration
import random
import pandas

from sklearn.model_selection import cross_val_score

random.seed(2020)

# 3 Experiments: (R1-R3)
#   -Test for 3 different models: SVM  

#Potential other models: Linear & Polynomial Regression, Clustering

def readAndAppendData(name):
  file1 = open(name, "r")  
  if name == "Difficulty":
    for i in range(100):
      num1 = int(file1.readline())
      num2 = int(file1.readline())
      num3 = int(file1.readline())
      num4 = int(file1.readline())
      output = int(file1.readline())
      inputList = [num1, num2, num3, num4]
      input_data1.append(inputList)
      output_data1.append(output)
  elif name == "Order":
    for i in range(100):
      num1 = int(file1.readline())
      num2 = int(file1.readline())
      num3 = int(file1.readline())
      num4 = int(file1.readline())
      num5 = int(file1.readline())
      num6 = int(file1.readline())
      num7 = int(file1.readline())
      num8 = int(file1.readline())
      output = int(file1.readline())
      inputList = [num1, num2, num3, num4, num5, num6, num7, num8]
      input_data2.append(inputList)
      output_data2.append(output)
  elif name == "Type":
    for i in range(100):
      num1 = int(file1.readline())
      num2 = int(file1.readline())
      num3 = int(file1.readline())
      num4 = int(file1.readline())
      output = int(file1.readline())
      inputList = [num1, num2, num3, num4]
      input_data3.append(inputList)
      output_data3.append(output)
  file1.close()

def addLists(sumList, newList):
  returnList = [0, 0, 0, 0, 0]
  for i in range(len(sumList)):
    if newList[i] > 0:
      returnList[i] = sumList[i] + newList[i]

  return returnList

def avgList(sumList):
  sum = 0
  for num in sumList:
    sum = sum + num
  sum = sum/len(sumList)

  return sum

#Experiment 1: SVM

#R1: Difficulty Distribution
input_data1 = [
  [25, 25, 25, 25],
  [10, 10, 40, 40],
  [20, 20, 30, 30],
]

output_data1 = [
  80,
  78,
  90,
]

#R2: Order
input_data2 = [
  [1, 2, 3, 4, 1, 2, 3, 4],
  [1, 1, 2, 2, 3, 3, 4, 4],
  [2, 4, 3, 4, 1, 2, 3, 1]
]

output_data2 = [
  90,
  80,
  85
]

#R3: Type Distribution
input_data3 = [
  [25, 25, 25, 25],
  [30, 30, 20, 20],
  [85, 5, 5, 5]
]

output_data3 = [
  70,
  84,
  92
]

dataGeneration.generateDummyData("Difficulty")
readAndAppendData("Difficulty")

dataGeneration.generateDummyData("Order")
readAndAppendData("Order")

dataGeneration.generateDummyData("Type")
readAndAppendData("Type")

svmExp1 = []
svmExp2 = []
svmExp3 = []
lrExp1 = []
lrExp2 = []
lrExp3 = []
prExp1 = []
prExp2 = []
prExp3 = []

CV = 5

newFile = open("finalResult", "a")
SVMScore = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
LRScore = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]
PRScore = [[0,0,0,0,0], [0,0,0,0,0], [0,0,0,0,0]]


#Running full experimentation N times
for i in range(50):
  print("Result: ", i + 1)
  newFile.write("Result: " + str(i + 1) + "\n")
  #Generate Difficulty Predictor
  num1 = random.randint(0, 100)
  num2 = random.randint(0, 100)
  while num1 + num2 > 100:
    num2 = random.randint(0, 100 - num1)
  num3 = random.randint(0, 100)
  while num1 + num2 + num3 > 100:
    num3 = random.randint(0, 100 - num1 - num2)
  num4 = 100 - num1 - num2 - num3
  difficultyPred = [num1,num2,num3,num4]  #ie [15,30,15,40]
  
  #Generate Order Predictor
  orderPred = []                          #ie [4,3,2,1,1,2,3,4]
  for i in range(8): 
    orderPred.append(random.randint(1,4))
  
  #Generate Type Predictor
  num1 = random.randint(0, 100)
  num2 = random.randint(0, 100)
  while num1 + num2 > 100:
    num2 = random.randint(0, 100 - num1)
  num3 = random.randint(0, 100)
  while num1 + num2 + num3 > 100:
    num3 = random.randint(0, 100 - num1 - num2)
  num4 = 100 - num1 - num2 - num3
  typePred = [num1,num2,num3,num4]                #ie [30,30,20,20]

  print("\nDifficulty: ", difficultyPred)
  newFile.write("\nDifficulty: " + str(difficultyPred))
  print("Order: ", orderPred)
  newFile.write("\nOrder: " + str(orderPred))
  print("Type: ", typePred)
  newFile.write("\nType: " + str(typePred))

  #__SVM__
  print("____Testing with SVM____")
  newFile.write("\n\n____Testing with SVM____")
  model1 = svm.SVC()

  #R1
  model1.fit(input_data1,output_data1)
  print("Experiment 1: Difficulty Distribution")
  newFile.write("\nExperiment 1: Difficulty Distribution")
  var = model1.predict([difficultyPred])
  svmExp1.append(var[0])
  score = cross_val_score(model1, input_data1, output_data1, cv = CV)
  
  SVMScore[0] = addLists(SVMScore[0], score)
  
  print(var, "Score: ", score)
  newFile.write("\n" + str(var)+ "Score: "+ str(score))

  #100 input data, cv = 5, data--> (1/5)
  #20-20-20-20-20 input/output Data
  #newInputOutputData = X-20-20-20-20 (80)
  #model.predict(20) compares the result to the ACTUAL output

  # score = cross_val_score(model1, input_data1, output_data1, cv = 5)
  # print("Score: ", score)

  #R2
  model1.fit(input_data2, output_data2)
  print("Experiment 2: Order")
  newFile.write("\nExperiment 2: Order")
  var = model1.predict([orderPred])
  svmExp2.append(var[0])
  score = cross_val_score(model1, input_data2, output_data2, cv = CV)
  SVMScore[1] = addLists(SVMScore[1], score)
  print(var, "Score: ", score)
  newFile.write("\n" + str(var)+ "Score: "+ str(score))

  #R3
  model1.fit(input_data3, output_data3)
  print("Experiment 3: Type Distribution")
  newFile.write("\nExperiment 3: Type Distribution")
  var = model1.predict([typePred])
  svmExp3.append(var[0])
  score = cross_val_score(model1, input_data3, output_data3, cv = CV)
  SVMScore[2] = addLists(SVMScore[2], score)
  print(var, "Score: ", score)
  newFile.write("\n" + str(var)+ "Score: "+ str(score))

  #Linear Regression
  print("\n____Testing Linear Regression____")
  newFile.write("\n\n____Testing Linear Regression____")
  model2 = linear_model.LinearRegression()


  #R1
  model2.fit(input_data1,output_data1)
  print("Experiment 1: Difficulty Distribution")
  newFile.write("\nExperiment 1: Difficulty Distribution")
  var = model2.predict([difficultyPred])
  lrExp1.append(var[0])
  score = cross_val_score(model2, input_data1, output_data1, cv = CV)
  LRScore[0] = addLists(LRScore[0], score)
  print(var, "Score: ", score)
  newFile.write("\n" + str(var)+ "Score: "+ str(score))

  #R2
  model2.fit(input_data2, output_data2)
  print("Experiment 2: Order")
  newFile.write("\nExperiment 2: Order")
  var = model2.predict([orderPred])
  lrExp2.append(var[0])
  score = cross_val_score(model2, input_data2, output_data2, cv = CV)
  LRScore[1] = addLists(LRScore[1], score)
  print(var, "Score: ", score)
  newFile.write("\n" + str(var)+ "Score: "+ str(score))


  #R3
  model2.fit(input_data3, output_data3)
  print("Experiment 3: Type Distribution")
  newFile.write("\nExperiment 3: Type Distribution")
  var = model2.predict([typePred])
  lrExp3.append(var[0])
  score = cross_val_score(model2, input_data3, output_data3, cv = CV)
  LRScore[2] = addLists(LRScore[2], score)
  print(var, "Score: ", score)
  newFile.write("\n" + str(var)+ "Score: "+ str(score))

  #Polynomial Regression 
  print("\n____Testing Polynomial  Regression____")
  newFile.write("\n\n____Testing Polynomial  Regression____")
  model3 = make_pipeline(PolynomialFeatures(2), Ridge())


  #R1
  model3.fit(input_data1,output_data1)
  print("Experiment 1: Difficulty Distribution")
  newFile.write("\nExperiment 1: Difficulty Distribution")
  var = model3.predict([difficultyPred])
  prExp1.append(var[0])
  score = cross_val_score(model3, input_data1, output_data1, cv = CV)
  PRScore[0] = addLists(PRScore[0], score)
  print(var, "Score: ", score)
  newFile.write("\n" + str(var)+ "Score: "+ str(score))

  #R2
  model3.fit(input_data2, output_data2)
  print("Experiment 2: Order")
  newFile.write("\nExperiment 2: Order")
  var = model3.predict([orderPred])
  prExp2.append(var[0])
  score = cross_val_score(model3, input_data2, output_data2, cv = CV)
  PRScore[1] = addLists(PRScore[1], score)
  print(var, "Score: ", score)
  newFile.write("\n" + str(var)+ "Score: "+ str(score))

  #R3
  model3.fit(input_data3, output_data3)
  print("Experiment 3: Type Distribution")
  newFile.write("\nExperiment 3: Type Distribution")
  var = model3.predict([typePred])
  prExp3.append(var[0])
  score = cross_val_score(model3, input_data3, output_data3, cv = CV)
  PRScore[2] = addLists(PRScore[2], score)
  print(var, "Score: ", score)
  newFile.write("\n" + str(var)+ "Score: "+ str(score) + "\n\n")

#Saving results of all N experiments
results_output = pandas.DataFrame({
  "SVM Exp. 1" : svmExp1,
  "SVM Exp. 2" : svmExp2,
  "SVM Exp. 3" : svmExp3,
  "LR Exp. 1" : lrExp1,
  "LR Exp. 2" : lrExp2,
  "LR Exp. 3" : lrExp3,
  "PR Exp. 1" : prExp1,
  "PR Exp. 2" : prExp2,
  "PR Exp. 3" : prExp3
})
results_output.to_csv("results.csv", sep=",", mode = "a", index = False)
newFile.close()

# setup 3 different ML models
# training
# evaluation


# generate dummy data ? 

# compare the result with 3 algorithms 
#[5, 5, 30, 60]
#[1, 3, 2, 4, 1, 3, 2, 4]
#[10, 10, 25, 55]

#1. Run the experiments with a lot more data
#2. Save all of the experiment results
#3. Clean up our program

#Do this for all 3 models
#Linear Regression Score
#Exp 1:
#Exp 2:
#Exp 3: 
#Overall:


print("\nSVM Accuracy Score")
print("Experiment 1:", avgList(SVMScore[0]) / 50)
print("Experiment 2:", avgList(SVMScore[1]) / 50)
print("Experiment 3:", avgList(SVMScore[2]) / 50)
overall = 0
for i in range(len(SVMScore)):
 overall += avgList(SVMScore[i])
print("Overall: ", overall/150)

print("Linear Regression Accuracy Score")
print("Experiment 1:", avgList(LRScore[0]) / 50)
print("Experiment 2:", avgList(LRScore[1]) / 50)
print("Experiment 3:", avgList(LRScore[2]) / 50)
overall = 0
for i in range(len(LRScore)):
 overall += avgList(LRScore[i])
print("Overall: ", overall/150)

print("Polynomial Regression Accuracy Score")
print("Experiment 1:", avgList(PRScore[0]) / 50)
print("Experiment 2:", avgList(PRScore[1]) / 50)
print("Experiment 3:", avgList(PRScore[2]) / 50)
overall = 0
for i in range(len(PRScore)):
 overall += avgList(PRScore[i])
print("Overall: ", overall/150)


#Normalize the random seed - Done
#Fix the Accuracy bug - Done
#Set the Program to run for 500-1000 times
#Reorganize our final data

# SVM Accuracy Score
# Experiment 1: 0.049047619047619034
# Experiment 2: 0.10666666666666667
# Experiment 3: 0.03809523809523809
# Overall:  0.06460317460317459
# Linear Regression Accuracy Score
# Experiment 1: 0.41091873684448993
# Experiment 2: 0.1269050620188689
# Experiment 3: 0.18073520810255644
# Overall:  0.23951966898863844
# Polynomial Regression Accuracy Score
# Experiment 1: 0.36057486963063495
# Experiment 2: 0.11015501059159545
# Experiment 3: 0.1140803087920059
# Overall:  0.1949367296714121
```

Data Generation
```python
import random

random.seed(2020)

def generateDiffData():
  student = random.randint(1,4)
  if student == 1:
    #[20-30, 20-30, 20-30, v20-30]
    #[65 - 85]
    diffOne = random.randint(20, 30)
    diffTwo = random.randint(20, 30)
    diffThree = random.randint(20, 30)
    diffFour = 100 - (diffOne + diffTwo + diffThree)
    inputResult = [diffOne, diffTwo, diffThree, diffFour]
    outputResult = [random.randint(65, 85)]
    return inputResult, outputResult
    
    #return [20, 29, 22, 29], [77]
  elif student == 2:
    #[0-5, 0-5, 30-40, v40-60]
    #[50 - 75]
    diffOne = random.randint(0, 5)
    diffTwo = random.randint(0, 5)
    diffThree = random.randint(30, 40)
    diffFour = 100 - (diffOne + diffTwo + diffThree)
    inputResult = [diffOne, diffTwo, diffThree, diffFour]
    outputResult = [random.randint(50, 75)]
    return inputResult, outputResult
    
    #return [0, 1, 40, 59], [57]
  elif student == 3:
    #[5-10, 10-15, v45-65, 20-30]
    #[60 - 80]
    diffOne = random.randint(5, 10)
    diffTwo = random.randint(10, 15)
    diffFour = random.randint(20, 30)
    diffThree = 100 - (diffOne + diffTwo + diffFour)
    inputResult = [diffOne, diffTwo, diffThree, diffFour]
    outputResult = [random.randint(60, 80)]
    return inputResult, outputResult
    
    #return [8, 14, 48, 30], [63]
  else:
    #[20-30, 35-50, v0-35, 10-20]
    #[70-90]
    diffOne = random.randint(20, 30)
    diffTwo = random.randint(35, 50)
    diffFour = random.randint(10, 20)
    diffThree = 100 - (diffOne + diffTwo + diffFour)
    inputResult = [diffOne, diffTwo, diffThree, diffFour]
    outputResult = [random.randint(70, 90)]
    return inputResult, outputResult

    #return [22, 47, 13, 18], [81]
    #Student 4

#Students scores in Question Ordering
def generateOrderData():
  student = random.randint(1,4)
  if student == 1:
    inputResult = []
    for i in range(8):
      inputResult.append(random.randint(1,4))
    
    score = random.randint(50,100)
    return inputResult, [score]

  elif student == 2: 
    score = random.randint(80,90)
    return [1,2,3,4,1,2,3,4], [score]
  
  elif student == 3:
    score = random.randint(75,85)
    return [1,1,2,2,3,3,4,4], [score]
  else:
    score = random.randint(70,80)
    return [2,2,3,3,4,4,1,1], [score]

#Students scores in Type Distribution
def generateTypeData():
  student = random.randint(1,4)
  if student == 1:
    #[20-30, 20-30, 20-30, v20-30]
    #[55 - 70]
    typeOne = random.randint(20, 30)
    typeTwo = random.randint(20, 30)
    typeThree = random.randint(20, 30)
    typeFour = 100 - (typeOne + typeTwo + typeThree)
    inputResult = [typeOne, typeTwo, typeThree, typeFour]
    outputResult = [random.randint(55, 70)]
    return inputResult, outputResult
  elif student == 2:
    #[30-40, 30-40, v10-20, v10-20]
    #[65-80]
    typeOne = random.randint(30, 40)
    typeTwo = random.randint(30, 40)
    typeThree = int((100 - (typeOne + typeTwo)) / 2)
    typeFour = 100 - (typeOne + typeTwo + typeThree)
    inputResult = [typeOne, typeTwo, typeThree, typeFour]
    outputResult = [random.randint(65, 80)]
    return inputResult, outputResult
  elif student == 3:
    #70-91, v3-10, v3-10, v3-10]
    #[60-90]
    typeOne = random.randint(70, 91)
    typeTwo = int((100 - typeOne)/3)
    typeThree = typeTwo
    typeFour = 100 - (typeOne + typeTwo + typeThree)
    inputResult = [typeOne, typeTwo, typeThree, typeFour]
    outputResult = [random.randint(60, 90)]
    return inputResult, outputResult
  else: 
    #[v10-15, 35-40, 35-40, v10-15]
    #[65 - 85]
    typeTwo = random.randint(35, 40)
    typeThree = typeTwo
    typeOne = int((100 - (typeTwo + typeThree)) / 2)
    typeFour = 100 - (typeOne + typeTwo + typeThree)
    inputResult = [typeOne, typeTwo, typeThree, typeFour]
    outputResult = [random.randint(65, 85)]
    return inputResult, outputResult

def listToString(nums):
  newList = []
  for num in nums:
   newList.append(str(num) + "\n")
  return newList

def generateDummyData(name):
  file1 = open(name, "w")
  
  if name == "Difficulty":
    for i in range(100):
      var = generateDiffData()
      file1.writelines(listToString(var[0]))
      file1.write(str(var[1][0]) + "\n")
  elif name == "Order":
    for i in range(100):
      var = generateOrderData()
      file1.writelines(listToString(var[0]))
      file1.write(str(var[1][0]) + "\n")
  else:
    for i in range(100):
      var = generateTypeData()
      file1.writelines(listToString(var[0]))
      file1.write(str(var[1][0]) + "\n")

  file1.close()
```


##


