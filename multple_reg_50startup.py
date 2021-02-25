import pandas as pd
from sklearn.preprocessing import LabelEncoder
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv(r'E:\ExcelR ass\linear_reg\multi_linear_reg_dataset\50_Startups.csv')
# print(data.head())
# print(data.State.unique())


# convert categorical data :
le = LabelEncoder()
data["State"] = data["State"].astype('category')
data["State"] = le.fit_transform(data["State"])
data = data.rename({"R&D Spend":"rdspend", "Marketing Spend":"marketingspend"}, axis=1)
# print(data.head())
# print(data.info())

# print(data.corr())

# using ols :
model_1 = smf.ols('Profit~rdspend+Administration+marketingspend+State', data=data).fit()
# print(model_1.summary())           # R2 = 0.951 , AIC = 1061

model_2 = smf.ols('np.log(Profit)~rdspend+Administration+marketingspend+State', data=data).fit()
# print(model_2.summary())
# if take np.log(Profit) , R2 = 0.763, AIC = 1.843

model_3 = smf.ols('np.log(Profit)~rdspend+Administration+marketingspend', data=data).fit()
# print(model_3.summary())
# if dont take State then R2 = 0.762 , AIC = 0.1972

model_4 = smf.ols('np.log(Profit)~rdspend+marketingspend+State', data=data).fit()
# print(model_4.summary())
# if dont take administration then, R2 =0.763, AIC = -0.09070
model_5 = smf.ols('np.log(Profit)~rdspend+marketingspend', data=data).fit()
# print(model_5.summary())
# R2 = 0.761, AIC = -1.741



# Table for R2_value and AIC value :
model_df = pd.DataFrame({"models":["model_1", "model_2", "model_3", "model_4", "model_5"],
                         "R2_value":[0.951, 0.763, 0.762, 0.763, 0.761],
                         "AIC_value":[1061, 1.843, 0.1972, -0.09070, -1.741]})
# print(model_df)


# from above we can take model_4 for predictiong newdata :
new_data = pd.DataFrame({"rdspend":120000 , "marketingspend":300000, "State":2}, index=[0])
# print(np.exp(model_4.predict(new_data)))


# print(data.head())

# here we're working on model_4 :
x = data.iloc[:,0:4]
# print(x.head())
x = x.drop("Administration", axis=1)
# print(x.head())
coef = model_4.params[1:]
# print(coef)
coeff_df = pd.DataFrame(coef, x.columns, columns= ["coefficients"])
# print(coeff_df.coefficients[0])

# plotting of rdspend and marketing spend column :
# plt.scatter(x.rdspend, x.marketingspend, color="blue")
# plt.show()
# print(x.State.unique)



# plot predicted vs actual :
y_pred = np.exp(model_4.predict(x))
# print(data.shape)
y_actual = data.iloc[:,4]
# print(y_actual)
# plt.scatter(y_actual, y_pred)
# plt.xlabel("Actual Profit values")
# plt.ylabel("Predicted Profit values")
# plt.show()


# plot heatmap :
# sns.pairplot(data)
# sns.heatmap(data.corr(), annot=True)
# plt.show()

# Distribution graph using histogram :
# plt.hist(x.rdspend, color="red", alpha=0.5)
# plt.hist(x.marketingspend, color="blue", alpha=0.5)
# plt.hist(x.State, color="black", alpha=1)
# plt.show()


# plotting scatter point with respect to each state :
state_0 = x[x["State"] == 0]
state_1 = x[x["State"] == 1]
state_2 = x[x["State"] == 2]
# plt.scatter(x=state_0.rdspend, y=state_0.marketingspend, color="blue", alpha=0.5)
# plt.scatter(x=state_1.rdspend, y=state_1.marketingspend, color="red", alpha=0.5)
# plt.scatter(x=state_2.rdspend, y=state_2.marketingspend, color="green", alpha=0.5)
# plt.show()


# plot with respect to each state :
state_0_fit = np.polyfit(state_0.rdspend, state_0.marketingspend, 1)
state_1_fit = np.polyfit(state_1.rdspend, state_1.marketingspend, 1)
state_2_fit = np.polyfit(state_2.rdspend, state_2.marketingspend, 1)
# print(state_0_fit[0])
# sns.regplot(x=state_0.rdspend, y=state_0_fit[0]*state_0.rdspend + state_0_fit[1], label="New-York")
# sns.regplot(x=state_1.rdspend, y=state_1_fit[0]*state_1.rdspend + state_1_fit[1], label="California")
# sns.regplot(x=state_2.rdspend, y=state_2_fit[0]*state_2.rdspend + state_2_fit[1], label="Florida")
# plt.legend()
# plt.show()


# plotting line with points :
model_intercept = np.exp(model_4.params[0])
y_eq = model_intercept+ x.rdspend*coeff_df.coefficients[0] + x.marketingspend*coeff_df.coefficients[1] + x.State*coeff_df.coefficients[2]

# regression plot for each column with respect to final_equation line :
# sns.regplot(x=x.rdspend, y=y_eq, color="blue", label="R&Dspend")
# sns.regplot(x=x.marketingspend, y=y_eq, color="green", label="Marketingspend")
# sns.regplot(x=x.State, y=y_eq, color="orange", label="State")
# plt.legend()
# plt.show()



# regression plot for actual and preddicted :
# sns.regplot(x=y_actual, y=y_pred)
# plt.show()

# regression plot for actual, prediction and equation line :
# sns.regplot(x=y_actual, y=y_eq, label="actual profit values", color="blue")
# sns.regplot(x=y_pred, y=y_eq, label="predicted profit values", color="orange")
# plt.legend()
# plt.show()