@@ -0,0 +1,407 @@
# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:33.672424Z","iopub.execute_input":"2023-02-04T11:56:33.672935Z","iopub.status.idle":"2023-02-04T11:56:33.691842Z","shell.execute_reply.started":"2023-02-04T11:56:33.672895Z","shell.execute_reply":"2023-02-04T11:56:33.690159Z"}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [markdown]
# # Getting the dataset through pandas.

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:33.694329Z","iopub.execute_input":"2023-02-04T11:56:33.694840Z","iopub.status.idle":"2023-02-04T11:56:34.341344Z","shell.execute_reply.started":"2023-02-04T11:56:33.694806Z","shell.execute_reply":"2023-02-04T11:56:34.339886Z"}}
# credit = pd.read_csv("/kaggle/input/credit-score-data/train.csv")
credit = pd.read_csv("/data/train.csv")

# %% [markdown]
# # A peek at a small part of the complete dataset.

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:34.342864Z","iopub.execute_input":"2023-02-04T11:56:34.343271Z","iopub.status.idle":"2023-02-04T11:56:34.364414Z","shell.execute_reply.started":"2023-02-04T11:56:34.343240Z","shell.execute_reply":"2023-02-04T11:56:34.363044Z"}}
print(credit.head())

# %% [markdown]
# # Info of the columns that are present in our dataset

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:34.368040Z","iopub.execute_input":"2023-02-04T11:56:34.369137Z","iopub.status.idle":"2023-02-04T11:56:34.430183Z","shell.execute_reply.started":"2023-02-04T11:56:34.369072Z","shell.execute_reply":"2023-02-04T11:56:34.428853Z"}}
print(credit.info())

# %% [markdown]
# # Checking whether the dataset has any null values

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:34.432348Z","iopub.execute_input":"2023-02-04T11:56:34.433261Z","iopub.status.idle":"2023-02-04T11:56:34.479745Z","shell.execute_reply.started":"2023-02-04T11:56:34.433205Z","shell.execute_reply":"2023-02-04T11:56:34.478618Z"}}
print(credit.isnull().sum())

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:34.481337Z","iopub.execute_input":"2023-02-04T11:56:34.481994Z","iopub.status.idle":"2023-02-04T11:56:34.498595Z","shell.execute_reply.started":"2023-02-04T11:56:34.481926Z","shell.execute_reply":"2023-02-04T11:56:34.497212Z"}}
credit["Credit_Score"].value_counts()

# %% [markdown]
# # **Data Exploration**

# %% [markdown]
# # Exploring the  dataset as it has features that can train a Machine Learning model for credit score classification. 
# 

# %% [markdown]
# # Using "plotly" library to make interactive chartz for better understanding.

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:34.499932Z","iopub.execute_input":"2023-02-04T11:56:34.500620Z","iopub.status.idle":"2023-02-04T11:56:34.511638Z","shell.execute_reply.started":"2023-02-04T11:56:34.500583Z","shell.execute_reply":"2023-02-04T11:56:34.510420Z"}}
import plotly.express as px

# %% [markdown]
# # Exploring each feature to see if it affects the persons credit scores

# %% [markdown]
# # 1. Occupation

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:34.513126Z","iopub.execute_input":"2023-02-04T11:56:34.513635Z","iopub.status.idle":"2023-02-04T11:56:35.019102Z","shell.execute_reply.started":"2023-02-04T11:56:34.513593Z","shell.execute_reply":"2023-02-04T11:56:35.017477Z"}}
fig = px.box(credit, 
             x="Occupation",  
             color="Credit_Score", 
             title="Credit Scores Based on Occupation", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.show()

# %% [markdown]
# # 1. Result: not much difference

# %% [markdown]
# # 2. Annual Income:

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:35.023210Z","iopub.execute_input":"2023-02-04T11:56:35.024528Z","iopub.status.idle":"2023-02-04T11:56:35.527930Z","shell.execute_reply.started":"2023-02-04T11:56:35.024476Z","shell.execute_reply":"2023-02-04T11:56:35.525394Z"}}
fig = px.box(credit, 
             x="Credit_Score", 
             y="Annual_Income", 
             color="Credit_Score",
             title="Credit Scores Based on Annual Income", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

# %% [markdown]
# # 2. Result: More Annual Income -> Better credit score 

# %% [markdown]
# # 3. Monthly in-hand salary:

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:35.530413Z","iopub.execute_input":"2023-02-04T11:56:35.530991Z","iopub.status.idle":"2023-02-04T11:56:36.099240Z","shell.execute_reply.started":"2023-02-04T11:56:35.530919Z","shell.execute_reply":"2023-02-04T11:56:36.096624Z"}}
fig = px.box(credit, 
             x="Credit_Score", 
             y="Monthly_Inhand_Salary", 
             color="Credit_Score",
             title="Credit Scores Based on Monthly Inhand Salary", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

# %% [markdown]
# # 3. Result: More monthly in-hand salary -> better credit score

# %% [markdown]
# # 4. Number of Bank accounts:

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:36.101290Z","iopub.execute_input":"2023-02-04T11:56:36.101731Z","iopub.status.idle":"2023-02-04T11:56:36.600333Z","shell.execute_reply.started":"2023-02-04T11:56:36.101695Z","shell.execute_reply":"2023-02-04T11:56:36.598892Z"}}
fig = px.box(credit, 
             x="Credit_Score", 
             y="Num_Bank_Accounts", 
             color="Credit_Score",
             title="Credit Scores Based on Number of Bank Accounts", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

# %% [markdown]
# # 4 Conclusion: more bank accounts -> does not better credit score
# 

# %% [markdown]
# # 5. Number of credit cards

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:36.601846Z","iopub.execute_input":"2023-02-04T11:56:36.602214Z","iopub.status.idle":"2023-02-04T11:56:37.090392Z","shell.execute_reply.started":"2023-02-04T11:56:36.602182Z","shell.execute_reply":"2023-02-04T11:56:37.089027Z"}}
fig = px.box(credit, 
             x="Credit_Score", 
             y="Num_Credit_Card", 
             color="Credit_Score",
             title="Credit Scores Based on Number of Credit cards", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

# %% [markdown]
# # 5 Conclusion: More credit cards -> does not better credit score

# %% [markdown]
# # 6 average interest on loans and EMIs

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:37.092349Z","iopub.execute_input":"2023-02-04T11:56:37.092709Z","iopub.status.idle":"2023-02-04T11:56:37.577370Z","shell.execute_reply.started":"2023-02-04T11:56:37.092677Z","shell.execute_reply":"2023-02-04T11:56:37.576073Z"}}
fig = px.box(credit, 
             x="Credit_Score", 
             y="Interest_Rate", 
             color="Credit_Score",
             title="Credit Scores Based on the Average Interest rates", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

# %% [markdown]
# # 6 Conclusion: 
# # Average interest rate is 4 – 11%    -> Good credit score 
# # Average interest rate more than 15% -> Bad credit scores

# %% [markdown]
# # 7. Number of loans

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:37.579331Z","iopub.execute_input":"2023-02-04T11:56:37.580408Z","iopub.status.idle":"2023-02-04T11:56:38.062006Z","shell.execute_reply.started":"2023-02-04T11:56:37.580367Z","shell.execute_reply":"2023-02-04T11:56:38.059866Z"}}
fig = px.box(credit, 
             x="Credit_Score", 
             y="Num_of_Loan", 
             color="Credit_Score", 
             title="Credit Scores Based on Number of Loans Taken by the Person",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

# %% [markdown]
# # 7 Conclusion: more than three loans at a time -> Bad credit scores

# %% [markdown]
# # 8a. Delaying payments

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:38.063832Z","iopub.execute_input":"2023-02-04T11:56:38.064250Z","iopub.status.idle":"2023-02-04T11:56:38.570198Z","shell.execute_reply.started":"2023-02-04T11:56:38.064213Z","shell.execute_reply":"2023-02-04T11:56:38.568492Z"}}
fig = px.box(credit, 
             x="Credit_Score", 
             y="Delay_from_due_date", 
             color="Credit_Score",
             title="Credit Scores Based on Average Number of Days Delayed for Credit card Payments", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

# %% [markdown]
# # 8a Conclusion: delaying more than 12 payments from the due date -> Bad credit scores 

# %% [markdown]
# # 8b. frequently delaying payments

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:38.572664Z","iopub.execute_input":"2023-02-04T11:56:38.573154Z","iopub.status.idle":"2023-02-04T11:56:39.091746Z","shell.execute_reply.started":"2023-02-04T11:56:38.573111Z","shell.execute_reply":"2023-02-04T11:56:39.090092Z"}}
fig = px.box(credit, 
             x="Credit_Score", 
             y="Num_of_Delayed_Payment", 
             color="Credit_Score", 
             title="Credit Scores Based on Number of Delayed Payments",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

# %% [markdown]
# # 8b. Conclusion: 
# # delaying 4 – 12 payments -> No effect on credit Score 
# # more than 12 payments from the due date -> Bad Effect credit scores 
# 

# %% [markdown]
# # 9. More Debt

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:39.093555Z","iopub.execute_input":"2023-02-04T11:56:39.094054Z","iopub.status.idle":"2023-02-04T11:56:39.604443Z","shell.execute_reply.started":"2023-02-04T11:56:39.094011Z","shell.execute_reply":"2023-02-04T11:56:39.602823Z"}}
fig = px.box(credit, 
             x="Credit_Score", 
             y="Outstanding_Debt", 
             color="Credit_Score", 
             title="Credit Scores Based on Outstanding Debt",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

# %% [markdown]
# # 9 Conclusion: Debt of more than $1338 -> Bad credit scores 
#  

# %% [markdown]
# # 10. High Credit Utilization
# ***Credit utilization ratio = total debt / total available credit***

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:39.606101Z","iopub.execute_input":"2023-02-04T11:56:39.606446Z","iopub.status.idle":"2023-02-04T11:56:40.124406Z","shell.execute_reply.started":"2023-02-04T11:56:39.606415Z","shell.execute_reply":"2023-02-04T11:56:40.122747Z"}}
fig = px.box(credit, 
             x="Credit_Score", 
             y="Credit_Utilization_Ratio", 
             color="Credit_Score",
             title="Credit Scores Based on Credit Utilization Ratio", 
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

# %% [markdown]
# # 10 Conclusion: credit utilization -> No effect on credit scores

# %% [markdown]
# # 11. Credit History Age

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:40.126491Z","iopub.execute_input":"2023-02-04T11:56:40.126942Z","iopub.status.idle":"2023-02-04T11:56:40.638048Z","shell.execute_reply.started":"2023-02-04T11:56:40.126903Z","shell.execute_reply":"2023-02-04T11:56:40.636848Z"}}
fig = px.box(credit, 
             x="Credit_Score", 
             y="Credit_History_Age", 
             color="Credit_Score", 
             title="Credit Scores Based on Credit History Age",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

# %% [markdown]
# # 11 Conclusion: long credit history -> Good Credit Score  

# %% [markdown]
# # 12. Number of EMIs

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:40.639512Z","iopub.execute_input":"2023-02-04T11:56:40.640804Z","iopub.status.idle":"2023-02-04T11:56:41.155703Z","shell.execute_reply.started":"2023-02-04T11:56:40.640758Z","shell.execute_reply":"2023-02-04T11:56:41.153983Z"}}
fig = px.box(credit, 
             x="Credit_Score", 
             y="Total_EMI_per_month", 
             color="Credit_Score", 
             title="Credit Scores Based on Total Number of EMIs per Month",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

# %% [markdown]
# # 12 Conclusion: Number of EMIs -> No effect on credit score

# %% [markdown]
# # 13. Minthly Investments

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:41.158066Z","iopub.execute_input":"2023-02-04T11:56:41.158488Z","iopub.status.idle":"2023-02-04T11:56:41.682790Z","shell.execute_reply.started":"2023-02-04T11:56:41.158450Z","shell.execute_reply":"2023-02-04T11:56:41.681662Z"}}
fig = px.box(credit, 
             x="Credit_Score", 
             y="Amount_invested_monthly", 
             color="Credit_Score", 
             title="Credit Scores Based on Amount Invested Monthly",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

# %% [markdown]
# # 13 Conclusion: Monthly Investment -> No effect on credit score

# %% [markdown]
# # 14 Low Monthly Balance 

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:41.684125Z","iopub.execute_input":"2023-02-04T11:56:41.685243Z","iopub.status.idle":"2023-02-04T11:56:42.196484Z","shell.execute_reply.started":"2023-02-04T11:56:41.685173Z","shell.execute_reply":"2023-02-04T11:56:42.193207Z"}}
fig = px.box(credit, 
             x="Credit_Score", 
             y="Monthly_Balance", 
             color="Credit_Score", 
             title="Credit Scores Based on Monthly Balance Left",
             color_discrete_map={'Poor':'red',
                                 'Standard':'yellow',
                                 'Good':'green'})
fig.update_traces(quartilemethod="exclusive")
fig.show()

# %% [markdown]
# #  14 Conclusion: High monthly balance -> good for credit scores
# #  monthly balance less than $250 -> bad for credit scores

# %% [markdown]
# # **Credit Score Classification Model**

# %% [markdown]
# # One more important feature (Credit Mix) in the dataset is valuable for determining credit scores. 
# # The credit mix feature tells about the types of credits and loans you have taken.
# # As the Credit_Mix column is categorical, transform it into a numerical feature so that we can use it to train a Machine Learning model for the task of credit score classification:

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:42.198470Z","iopub.execute_input":"2023-02-04T11:56:42.199195Z","iopub.status.idle":"2023-02-04T11:56:42.223131Z","shell.execute_reply.started":"2023-02-04T11:56:42.199150Z","shell.execute_reply":"2023-02-04T11:56:42.221417Z"}}
credit["Credit_Mix"] = credit["Credit_Mix"].map({"Standard": 1, 
                               "Good": 2, 
                               "Bad": 0})

# %% [markdown]
# # split the data into features and labels by selecting the features important for the model

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:42.228114Z","iopub.execute_input":"2023-02-04T11:56:42.229181Z","iopub.status.idle":"2023-02-04T11:56:42.235136Z","shell.execute_reply.started":"2023-02-04T11:56:42.229129Z","shell.execute_reply":"2023-02-04T11:56:42.233572Z"}}
from sklearn.model_selection import train_test_split

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:42.237195Z","iopub.execute_input":"2023-02-04T11:56:42.238002Z","iopub.status.idle":"2023-02-04T11:56:42.269566Z","shell.execute_reply.started":"2023-02-04T11:56:42.237935Z","shell.execute_reply":"2023-02-04T11:56:42.267903Z"}}
x = np.array(credit[["Annual_Income", "Monthly_Inhand_Salary", 
                   "Num_Bank_Accounts", "Num_Credit_Card", 
                   "Interest_Rate", "Num_of_Loan", 
                   "Delay_from_due_date", "Num_of_Delayed_Payment", 
                   "Credit_Mix", "Outstanding_Debt", 
                   "Credit_History_Age", "Monthly_Balance"]])
y = np.array(credit[["Credit_Score"]])

# %% [markdown]
# # 1. split the data into training and test sets 
# # 2. train a credit score classification model

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:56:42.271611Z","iopub.execute_input":"2023-02-04T11:56:42.272124Z","iopub.status.idle":"2023-02-04T11:57:02.587304Z","shell.execute_reply.started":"2023-02-04T11:56:42.272086Z","shell.execute_reply":"2023-02-04T11:57:02.586444Z"}}
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                    test_size=0.33, 
                                                    random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtrain, ytrain)

# %% [markdown]
# # make predictions from the model by giving inputs to it according to the features used to train the model

# %% [code] {"execution":{"iopub.status.busy":"2023-02-04T11:59:25.489294Z","iopub.execute_input":"2023-02-04T11:59:25.490110Z","iopub.status.idle":"2023-02-04T12:00:30.335888Z","shell.execute_reply.started":"2023-02-04T11:59:25.490071Z","shell.execute_reply":"2023-02-04T12:00:30.334615Z"}}
#add the values by yourself to check the credit score:
print("Credit Score Prediction : ")
a = float(input("Annual Income: "))
b = float(input("Monthly Inhand Salary: "))
c = float(input("Number of Bank Accounts: "))
d = float(input("Number of Credit cards: "))
e = float(input("Interest rate: "))
f = float(input("Number of Loans: "))
g = float(input("Average number of days delayed by the person: "))
h = float(input("Number of delayed payments: "))
i = input("Credit Mix (Bad: 0, Standard: 1, Good: 3) : ")
j = float(input("Outstanding Debt: "))
k = float(input("Credit History Age: "))
l = float(input("Monthly Balance: "))

features = np.array([[a, b, c, d, e, f, g, h, i, j, k, l]])
print("Predicted Credit Score = ", model.predict(features))

# %% [code]


# %% [markdown] {"_kg_hide-output":true}
# 
