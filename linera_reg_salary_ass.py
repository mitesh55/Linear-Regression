import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# load the dataset :
data = pd.read_csv(r'E:\ExcelR ass\linear_reg\dataset\Salary_Data.csv')


# split the dataset into feature and label
# here we want to predict salary based on years of experience, so y will be label and x will be feature
x = data.iloc[:,0]
y = data.iloc[:,1]

"""
after spliting data visulize data using matplotlib or seaborn 

"""
# plt.scatter(x, y)
# sns.scatterplot(x=x, y=y, data=data)
# plt.hist(y)

# plt.show()


r2 = data.corr()
# print(r2)                 # here r2 = 0.9782, that means our data has strong and positive relation.

# make model for prediction :
model = smf.ols('Salary~YearsExperience', data=data).fit()
# print(model.summary())
# here R-squared = 0.957, Intercept = 2.579e+04 , YearsExperience = 9449.9623

# predict new_data :
new_data = pd.DataFrame({"YearsExperience":4}, index=[0])
print(model.predict(new_data))