"""
__author__: "Trisha P Malhotra (tpm6421)"
"""


import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn as skl
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge


"""
Taking original and test data as input 
"""
data_set = pd.read_csv("adult_data.txt",header=0)
test_set = pd.read_csv("adult_test.txt",header=0)

data_set_matrix = np.array(data_set)
data_set_matrix_test = np.array(test_set)

"""
Selecting columns
"""
marital_status_data = data_set_matrix[:,[5]]
marital_status_test = data_set_matrix_test[:,[5]]

employment_data = data_set_matrix[:,[1]]
employment_test = data_set_matrix_test[:,[1]]

region_data = data_set_matrix[:,[13]]
region_test = data_set_matrix_test[:,[13]]

occupation_data = data_set_matrix[:,[6]]
occupation_test = data_set_matrix_test[:,[6]]

sex_data = data_set_matrix[:,[9]]
sex_test = data_set_matrix_test[:,[9]]

race_data = data_set_matrix[:,[8]]
race_test = data_set_matrix_test[:,[8]]

relationship_data = data_set_matrix[:,[7]]
relationship_test = data_set_matrix_test[:,[7]]

asia = [" Cambodia", " China", " Hong", " India", " Iran", " Japan", " Laos"," Philippines", " Taiwan", " Thailand", " Vietnam"]
north_america =[" Canada", " Cuba", " Dominican-Republic", " El-Salvador", " Guatemala"," Haiti", " Honduras", " Jamaica ", " Mexico", " Nicaragua"," Outlying-US(Guam-USVI-etc)", " Puerto-Rico", " Trinadad&Tobago"," United-States"]
europe=[" England", " France", " Germany", " Greece", " Holand-Netherlands"," Hungary", " Ireland", " Italy", " Poland", " Portugal", " Scotland"," Yugoslavia"]
south_america =[" Columbia", " Ecuador", " Peru"]

"""
Converting categories into continuous data
"""
# original data
for i in range(0,len(region_data)):
    value = region_data[i]

    if value in north_america:
        value = 1
    elif value in asia:
        value = 2
    elif value in south_america:
        value = 3
    elif value in europe:
        value = 4
    else:
        value = 5
    region_data[i] = value

# test data
for i in range(0,len(region_test)):
    value = region_test[i]

    if value in north_america:
        value = 1
    elif value in asia:
        value = 2
    elif value in south_america:
        value = 3
    elif value in europe:
        value = 4
    else:
        value = 5
    region_test[i] = value

unemployed= [" Without-pay"," Never-worked"]
government_employed =[" State-gov"," Local-gov"," Federal-gov"]
self_employed = [" Self-emp-inc"," Self-emp-not-inc"]
private = [" Private"]

for i in range(0,len(employment_data)):
    value = employment_data[i]

    if value in unemployed:
        value = 1
    elif value in government_employed:
        value = 2
    elif value in self_employed:
        value = 3
    elif value in private:
        value = 4
    else:
        value = 5
    employment_data[i] = value


for i in range(0,len(employment_test)):
    value = employment_test[i]

    if value in unemployed:
        value = 1
    elif value in government_employed:
        value = 2
    elif value in self_employed:
        value = 3
    elif value in private:
        value = 4
    else:
        value = 5
    employment_test[i] = value


married = [" Married-AF-spouse"," Married-civ-spouse"," Married-spouse-absent"]
not_married =[" Divorced"," Separated"," Widowed"]

for i in range(0,len(marital_status_data)):
    value = marital_status_data[i]

    if value in married:
        value = 1
    elif value in not_married:
        value = 2
    else:

        value = 3
    marital_status_data[i] = value


for i in range(0,len(marital_status_test)):
    value = marital_status_test[i]

    if value in married:
        value = 1
    elif value in not_married:
        value = 2
    else:

        value = 3
    marital_status_test[i] = value

"""
Relabeling the output features to -1 for <=50K and +1 for >50K
"""
salary_data = data_set_matrix[:,[14]]
for i in range(0,len(salary_data)):
    value = salary_data[i]

    if value == " <=50K":
        value = 1
    elif value == " >50K":
        value = -1
    else:

        value = 3
    salary_data[i] = value

salary_test = data_set_matrix_test[:,[14]]
for i in range(0,len(salary_test)):
    value = salary_test[i]

    if value == " <=50K":
        value = 1
    elif value == " >50K":
        value = -1
    else:

        value = 3
    salary_test[i] = value

for i in range(0,len(sex_data)):
    value = sex_data[i]

    if value == " Male":
        value = 1
    elif value == " Female":
        value = 2
    else:

        value = 3
    sex_data[i] = value
for i in range(0,len(sex_test)):
    value = sex_test[i]

    if value == " Male":
        value = 1
    elif value == " Female":
        value = 2
    else:

        value = 3
    sex_test[i] = value
for i in range(0,len(race_data)):
    value = race_data[i]

    if value == "  White":
        value = 1
    elif value == "  Black":
        value = 2
    elif value == "   Asian-Pac-Islander":
        value = 3
    elif value == "   Amer-Indian-Eskimo":
        value = 4
    else:

        value = 5
    race_data[i] = value
for i in range(0,len(race_test)):
    value = race_test[i]

    if value == " White":
        value = 1
    elif value == " Black":
        value = 2
    elif value == " Asian-Pac-Islander":
        value = 3
    elif value == " Amer-Indian-Eskimo":
        value = 4
    else:

        value = 5
    race_test[i] = value
for i in range(0,len(relationship_data)):
    value = relationship_data[i]

    if value == " Not-in-family":
        value = 1
    elif value == " Husband":
        value = 2
    elif value == " Wife":
        value = 3
    elif value == " Own-child":
        value = 4
    elif value == " Unmarried":
        value = 5
    else:

        value = 6
    relationship_data[i] = value


for i in range(0,len(relationship_test)):
    value = relationship_test[i]

    if value == " Not-in-family":
        value = 1
    elif value == " Husband":
        value = 2
    elif value == " Wife":
        value = 3
    elif value == " Own-child":
        value = 4
    elif value == " Unmarried":
        value = 5
    else:

        value = 6
    relationship_test[i] = value

for i in range(0,len(occupation_data)):
    value = occupation_data[i]

    if value == " Adm-clerical":
        value = 1
    elif value == " Exec-managerial":
        value = 2
    elif value == " Handlers-cleaners":
        value = 3
    elif value == " Prof-specialty":
        value = 4
    elif value == " Other-service":
        value = 5
    elif value == " Sales":
        value = 6
    elif value == " Craft-repair":
        value = 7
    elif value == " Transport-moving":
        value = 8
    elif value == " Farming-fishing":
        value = 9
    elif value == " Machine-op-inspct":
        value = 10
    elif value == " Tech-support":
        value = 11
    elif value == " Protective-serv":
        value = 12
    elif value == " Armed-Forces":
        value = 13
    elif value == " Priv-house-serv":
        value = 14
    else:

        value = 15
    occupation_data[i] = value
for i in range(0,len(occupation_test)):
    value = occupation_test[i]

    if value == " Adm-clerical":
        value = 1
    elif value == " Exec-managerial":
        value = 2
    elif value == " Handlers-cleaners":
        value = 3
    elif value == " Prof-specialty":
        value = 4
    elif value == " Other-service":
        value = 5
    elif value == " Sales":
        value = 6
    elif value == " Craft-repair":
        value = 7
    elif value == " Transport-moving":
        value = 8
    elif value == " Farming-fishing":
        value = 9
    elif value == " Machine-op-inspct":
        value = 10
    elif value == " Tech-support":
        value = 11
    elif value == " Protective-serv":
        value = 12
    elif value == " Armed-Forces":
        value = 13
    elif value == " Priv-house-serv":
        value = 14
    else:

        value = 15
    occupation_test[i] = value

smaller_data_matrix = data_set_matrix
smaller_data_matrix[:,[13]] = region_data
smaller_data_matrix[:,[1]] = employment_data
smaller_data_matrix[:,[5]] = marital_status_data
smaller_data_matrix[:,[14]] = salary_data
smaller_data_matrix[:,[9]] = sex_data
smaller_data_matrix[:,[6]] = occupation_data
smaller_data_matrix[:,[7]] = relationship_data
smaller_data_matrix[:,[8]] = race_data
smaller_matrix= smaller_data_matrix[:,[0,1,2,4,5,6,7,8,9,10,11,12,13]]
smaller_data_matrix_test = data_set_matrix_test
smaller_data_matrix_test[:,[13]] = region_test
smaller_data_matrix_test[:,[1]] = employment_test
smaller_data_matrix_test[:,[5]] = marital_status_test
smaller_data_matrix_test[:,[14]] = salary_test
smaller_data_matrix_test[:,[9]] = sex_test
smaller_data_matrix_test[:,[6]] = occupation_test
smaller_data_matrix_test[:,[7]] = relationship_test
smaller_data_matrix_test[:,[8]] = race_test
smaller_matrix_test= smaller_data_matrix_test[:,[0,1,2,4,5,6,7,8,9,10,11,12,13]]


# Linear Regression
regression= skl.linear_model.LinearRegression()
regression.fit(smaller_matrix,salary_data)
prediction = regression.predict(smaller_matrix_test)


print("Coefficients of linear regression model ")
print(regression.coef_)

print("Mean squared error: %f"
      % mean_squared_error(salary_test, prediction))

alpha_value =[1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10,100]

ridge_coeffs=[]

ridge_error = []

# Ridge regression with different alpha values
for alpha in alpha_value:
    ridgeReg = Ridge(alpha=alpha, normalize=True)

    ridgeReg.fit(smaller_matrix, salary_data)

    ridge_pred = ridgeReg.predict(smaller_matrix_test)

    print("Coefficients of ridge regression model, with alpha = "+str(alpha))
    print(ridge_coeffs)

    print("Mean squared error: %f"
          % mean_squared_error(salary_test, ridge_pred))
    ridge_coeffs.append(ridgeReg.coef_)
    ridge_error.append(mean_squared_error(salary_test, ridge_pred))

# lasso with different alpha values

lasso_coeffs = []

lasso_error = []
for alpha in alpha_value:
    lasso = skl.linear_model.Lasso(alpha=alpha, normalize=True)

    lasso.fit(smaller_matrix, salary_data)

    pred_lasso = lasso.predict(smaller_matrix_test)

    print("Coefficients of Lasso regression model, with alpha =  "+str(alpha))

    print("Mean squared error: %f"
          % mean_squared_error(salary_test, pred_lasso))
    lasso_coeffs.append(lasso.coef_)
    lasso_error.append(mean_squared_error(salary_test, pred_lasso))

# classification

#  logistic regression, perceptron, linear support vector machine

# logistic regression

log_regr= skl.linear_model.LogisticRegression()
log_regr.fit(smaller_matrix,(salary_data.ravel()).astype(int))
log_regr_pred = log_regr.predict(smaller_matrix_test)
print("Accuracy score for Logistic Regression: ")
print(skl.metrics.accuracy_score((salary_test.ravel()).astype(int), log_regr_pred))
print("Precision, Recall, F1 score score for Logistic Regression: ")
print(skl.metrics.classification_report((salary_test.ravel()).astype(int),log_regr_pred))


# perceptron
perceptron = skl.linear_model.Perceptron(n_iter=20)
perceptron.fit(smaller_matrix,(salary_data.ravel()).astype(int))
perceptron_pred = perceptron.predict(smaller_matrix_test)
print("Accuracy score for Perceptron:")
print(skl.metrics.accuracy_score((salary_test.ravel()).astype(int), perceptron_pred))

print("Precision, Recall, F1 score score for Perceptron: ")
print(skl.metrics.classification_report((salary_test.ravel()).astype(int),perceptron_pred))

#support vector machine
svm_model= skl.svm.SVC()
svm_model.fit(smaller_matrix,(salary_data.ravel()).astype(int))
svm_model_pred = log_regr.predict(smaller_matrix_test)
print("Accuracy score for SVM model:")
print(skl.metrics.accuracy_score((salary_test.ravel()).astype(int), svm_model_pred))
print("Precision, recall, F1 score score for SVC:")
print(skl.metrics.classification_report((salary_test.ravel()).astype(int),svm_model_pred))

# plotting ROC curves


# logistic regression

lrg_fpr,lrg_tpr,lrg_threshold = skl.metrics.roc_curve((salary_test.ravel()).astype(int),log_regr_pred)
plt.title('ROC curve for logistic regression')
plt.plot(lrg_fpr, lrg_tpr)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# perceptron
percp_fpr,percp_tpr,percp_threshold = skl.metrics.roc_curve((salary_test.ravel()).astype(int),perceptron_pred)
plt.title('ROC curve for Perceptron')
plt.plot(percp_fpr, percp_tpr)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# SVM

svm_fpr,svm_tpr,svm_threshold = skl.metrics.roc_curve((salary_test.ravel()).astype(int),svm_model_pred)
plt.title('ROC curve for SVC')
plt.plot(svm_fpr, svm_tpr)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# AUC
print(" Area under curve for logistic regression model:")
print(skl.metrics.auc(lrg_fpr, lrg_tpr))
print(" Area under curve for Perceptron model: ")
print(skl.metrics.auc(svm_fpr, svm_tpr))
print(" Area under curve for SVC model:")
print(skl.metrics.auc(percp_fpr, percp_tpr))