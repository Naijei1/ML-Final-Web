import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import streamlit as st

# --- Page Setup ---
st.set_page_config(page_title="Diabetes Predictor", layout="centered")
st.title("Diabetes Predictor")
st.text("INFO 1998: Introduction to Machine Learning – Final Project")
st.text("This webpage outlines our process for creating a diabetes predictor.")

# --- Load Data ---
df = pd.read_csv('diabetes.csv')
original = pd.read_csv('diabetes.csv')

# --- Dataset Overview ---
st.header("Dataset Overview")
st.markdown("Source: [Diabetes Dataset on Kaggle](https://www.kaggle.com/datasets/saurabh00007/diabetescsv)")
st.dataframe(df, use_container_width=True)

st.subheader("Statistical Summary")
st.dataframe(df.describe(), use_container_width=True)

# --- Column Selector ---
st.header("Column Viewer")
selected_columns = st.multiselect("Select columns to display", df.columns.tolist())

if selected_columns:
    st.subheader("Selected Data")
    st.dataframe(df[selected_columns], use_container_width=True)
else:
    st.info("Please select one or more columns to view the data.")

# --- Visualization Section Placeholder ---
st.header("Data Visualization")
st.markdown("We can visualize our data to better understand what we are working with.")
st.markdown("First, we visualize Frequency of Blood Pressure Values with a Positive and Negative Outcome")
fig, graph = plt.subplots()

outcome_positive = df.loc[df['Outcome'] == 1]
sns.kdeplot(outcome_positive['BloodPressure'], ax=graph)

graph.set_title('Frequency of Blood Pressure Values with a Positive Outcome')
graph.set_xlabel('Blood Pressure')
graph.set_ylabel('Frequency')

st.pyplot(fig)

fig, graph = plt.subplots()

outcome_negative = df.loc[df['Outcome'] == 0]

sns.kdeplot(outcome_negative['BloodPressure'], ax=graph)

graph.set_title('Frequency of Blood Pressure Values with a Positive Outcome')
graph.set_xlabel('Blood Pressure')
graph.set_ylabel('Frequency')

st.pyplot(fig)
st.info("""
It appears that both distributions have a peak at 0 mm Hg, which is highly unlikely for a living individual with a beating heart. 
This suggests that these data points are likely the result of faulty measurements or placeholders for missing blood pressure readings. 
We have a few options to address this: we could remove the bad data points from our dataset, 
but since other features might also contain inaccuracies, we want to be careful not to remove too much valuable data.
""")
st.write("instead of removing the data points we can try and replace the points with the average blood pressure values")
pos = df.loc[(df['Outcome'] == 1) & (df['BloodPressure'] > 0)]
st.write("Average blood pressure for positive outcomes: " + str(pos['BloodPressure'].mean()))

neg = df.loc[(df['Outcome'] == 0) & (df['BloodPressure'] > 0)]
st.write("Average blood pressure for negative outcomes: " + str(neg['BloodPressure'].mean()))

for x in range(len(df['BloodPressure'])):
    if df['BloodPressure'][x] == 0 & df['Outcome'][x] == 1:
        df['BloodPressure'][x] = 75.32142857142857
    if df['BloodPressure'][x] == 0 & df['Outcome'][x] == 0: 
        df['BloodPressure'][x] = 70.87733887733887
        
st.write('Looking at the first 10 data points, we see that the seventh data point has 0 for Blood Pressure and 0 for the outcome. After updating our data points, we expect the blood pressure for the seventh data point to be 70.87733887733887')
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Blood Pressure")
    st.dataframe(original.head(10)[["BloodPressure"]])

with col2:
    st.subheader("Edited Blood Pressure")
    st.dataframe(df.head(10)[["BloodPressure"]])
st.write('We repeat these steps for all columns in the dataset.')
st.write("### Glucose")
fig, graph = plt.subplots()

outcome_positive = df.loc[df['Outcome'] == 1]
outcome_negative = df.loc[df['Outcome'] == 0]

sns.kdeplot(outcome_positive['Glucose'], ax=graph)

graph.set_title('Frequency of Glucose Values with a Positive Outcome')
graph.set_xlabel('Glucose')
graph.set_ylabel('Frequency')

st.pyplot(fig)


fig, graph = plt.subplots()

sns.kdeplot(outcome_negative['Glucose'], ax=graph)

graph.set_title('Frequency of Glucose Values with a Negative Outcome')
graph.set_xlabel('Glucose')
graph.set_ylabel('Frequency')

# Display plot in Streamlit
st.pyplot(fig)
st.info("Again it seems like we have the same issue as Blood pressure: we will repeat our steps from above")
pos = df.loc[(df['Outcome'] == 1) & (df['Glucose'] > 0)]
st.write("Average Glucose values for positive outcomes: " + str(pos['Glucose'].mean()))

neg = df.loc[(df['Outcome'] == 0) & (df['Glucose'] > 0)]
st.write("Average Glucose values for negative outcomes: " + str(neg['Glucose'].mean()))

for x in range(len(df['Glucose'])):
    if df['Glucose'][x] == 0 & df['Outcome'][x] == 1:
        df['Glucose'][x] = 142.31954887218046
    if df['Glucose'][x] == 0 & df['Outcome'][x] == 0: 
        df['Glucose'][x] = 110.64386317907444

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Glucose Values")
    st.dataframe(original["Glucose"])

with col2:
    st.subheader("Edited Glucose Values")
    st.dataframe(df["Glucose"])
    
st.write("### SkinThickness")


fig, graph = plt.subplots()

outcome_positive = df.loc[df['Outcome'] == 1]
outcome_negative = df.loc[df['Outcome'] == 0]

sns.kdeplot(outcome_positive['SkinThickness'], ax=graph)

graph.set_title('Frequency of SkinThickness Values with a Positive Outcome')
graph.set_xlabel('SkinThickness')
graph.set_ylabel('Frequency')

st.pyplot(fig)


fig, graph = plt.subplots()

sns.kdeplot(outcome_negative['SkinThickness'], ax=graph)

graph.set_title('Frequency of SkinThickness Values with a Negative Outcome')
graph.set_xlabel('SkinThickness')
graph.set_ylabel('Frequency')

# Display plot in Streamlit
st.pyplot(fig)
st.warning("I'm pretty sure you cannot have a Skinthickness value of 0 so")
st.info("Again it seems like we have the same issue as Glucose and Blood pressure: we will repeat our steps from above")
pos = df.loc[(df['Outcome'] == 1) & (df['SkinThickness'] > 0)]
st.write("Average SkinThickness values for positive outcomes: " + str(pos['SkinThickness'].mean()))

neg = df.loc[(df['Outcome'] == 0) & (df['SkinThickness'] > 0)]
st.write("Average SkinThickness values for negative outcomes: " + str(neg['SkinThickness'].mean()))

for x in range(len(df['SkinThickness'])):
    if df['SkinThickness'][x] == 0 & df['Outcome'][x] == 1:
        df['SkinThickness'][x] = pos['SkinThickness'].mean()
    if df['SkinThickness'][x] == 0 & df['Outcome'][x] == 0: 
        df['SkinThickness'][x] = neg['SkinThickness'].mean()
        
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original SkinThickness")
    st.dataframe(original["SkinThickness"])

with col2:
    st.subheader("Edited SkinThickness")
    st.dataframe(df["SkinThickness"])
    
st.write("### BMI")


fig, graph = plt.subplots()

outcome_positive = df.loc[df['Outcome'] == 1]
outcome_negative = df.loc[df['Outcome'] == 0]

sns.kdeplot(outcome_positive['BMI'], ax=graph)

graph.set_title('Frequency of BMI Values with a Positive Outcome')
graph.set_xlabel('BMI')
graph.set_ylabel('Frequency')

st.pyplot(fig)


fig, graph = plt.subplots()

sns.kdeplot(outcome_negative['BMI'], ax=graph)

graph.set_title('Frequency of BMI Values with a Negative Outcome')
graph.set_xlabel('BMI')
graph.set_ylabel('Frequency')

# Display plot in Streamlit
st.pyplot(fig)
st.info("Again it seems like we have the same issue as Glucose and Blood pressure: we will repeat our steps from above")
pos = df.loc[(df['Outcome'] == 1) & (df['SkinThickness'] > 0)]
st.write("Average BMI values for positive outcomes: " + str(pos['BMI'].mean()))

neg = df.loc[(df['Outcome'] == 0) & (df['SkinThickness'] > 0)]
st.write("Average BMI values for negative outcomes: " + str(neg['BMI'].mean()))

for x in range(len(df['BMI'])):
    if df['BMI'][x] == 0 & df['Outcome'][x] == 1:
        df['BMI'][x] = pos['BMI'].mean()
    if df['BMI'][x] == 0 & df['Outcome'][x] == 0: 
        df['BMI'][x] = neg['BMI'].mean()
        
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original BMI")
    st.dataframe(original["BMI"])

with col2:
    st.subheader("Edited BMI")
    st.dataframe(df["BMI"])
    
st.write("### Heatmaps")
st.write("We can also visualize the relation of all of our data points with the outcome through a heat map.")
corr = df.corr()

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)

ax.set_title("Feature Correlation Heatmap")

st.pyplot(fig)
st.title("Models")
st.write("Now that we have fully cleaned up our data we can try a few different models to see which gives us a better predictive accuracy")

st.subheader("KNeighbors Classifier")


X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
Y = df['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

KNN = KNeighborsClassifier()
KNN.fit(X_train, Y_train)


predictions = KNN.predict(X_test)

st.code("""
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
Y = df['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

KNN = KNeighborsClassifier()
KNN.fit(X_train, Y_train)


predictions = KNN.predict(X_test)
        """)

st.write("Test Accuracy: ", accuracy_score(Y_test, predictions))
st.write("This is a good start but we can do better")

st.subheader("Decision Trees")
st.write("Instead of randomly guessing a good depth value we can check which depth gives us the best accuracy")
st.code("""
        best_depth = 1      
best_accuracy = 0   
for k in range(1, 100):
    model = tree.DecisionTreeClassifier(max_depth = k)
    model.fit(X_train, Y_train)

    pred_test = model.predict(X_test)
    acc_test = accuracy_score(pred_test, Y_test)

    if acc_test > best_accuracy :
        best_accuracy = acc_test
        best_depth = k
        """)

best_depth = 1      
best_accuracy = 0   
for k in range(1, 100):
    model = tree.DecisionTreeClassifier(max_depth = k)
    model.fit(X_train, Y_train)

    pred_test = model.predict(X_test)
    acc_test = accuracy_score(pred_test, Y_test)

    if acc_test > best_accuracy :
        best_accuracy = acc_test
        best_depth = k

st.write("Best accuracy: " + str(best_accuracy))
st.write("Best depth: " + str(best_depth))

st.write("We can now use this depth to our training and test accuracy")
st.code("""
        model = tree.DecisionTreeClassifier(max_depth = best_depth)

model.fit(X_train, Y_train)

dtree_pred_train = model.predict(X_train)
dtree_pred_test = model.predict(X_test)
        """)
model = tree.DecisionTreeClassifier(max_depth = best_depth)

model.fit(X_train, Y_train)

dtree_pred_train = model.predict(X_train)
dtree_pred_test = model.predict(X_test)
st.write("Train Accuracy: ", accuracy_score(dtree_pred_train, Y_train))
st.write("Test Accuracy: ", accuracy_score(dtree_pred_test, Y_test))
st.info("Unfortunately, our training accuracy is usually higher than our test accuracy, which implies that the model is overfitting to our training data points. We can try using a RandomForestClassifier, which uses multiple shallow trees to avoid overfitting. ")

st.subheader("RandomForestClassifier")
st.code("""
        from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
rf_model.fit(X_train, Y_train)


y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

accuracy_train = accuracy_score(Y_train, y_train_pred)
accuracy_test = accuracy_score(Y_test, y_test_pred)
        """)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)
rf_model.fit(X_train, Y_train)


y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

accuracy_train = accuracy_score(Y_train, y_train_pred)
accuracy_test = accuracy_score(Y_test, y_test_pred)
st.write("Train accuracy: ", accuracy_train)
st.write("Test accuracy: ", accuracy_test)
st.info("Again this does not seem much better but we can try one last thing")

st.subheader("Bayesian Optimization")
st.code("""
        import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score

import lightgbm as lgb

from hyperopt import fmin, rand, tpe, space_eval, STATUS_OK, Trials, hp
from hyperopt.pyll.stochastic import sample
#

params = {
    'max_depth': 5,
    'objective': 'binary',
    'metric': 'auc',
    'force_col_wise': True,
    'verbose': -1
}

train_data = lgb.Dataset(X_train, label=Y_train)
clf = lgb.train(params, train_data, num_boost_round=3)


y_train = clf.predict(X_train)
y_pred = clf.predict(X_test)

roc_auc2 = roc_auc_score(Y_train, y_train)
roc_auc = roc_auc_score(Y_test, y_pred)
        """)
from sklearn.metrics import accuracy_score, roc_auc_score

import lightgbm as lgb

from hyperopt import fmin, rand, tpe, space_eval, STATUS_OK, Trials, hp
from hyperopt.pyll.stochastic import sample
#

params = {
    'max_depth': 5,
    'objective': 'binary',
    'metric': 'auc',
    'force_col_wise': True,
    'verbose': -1
}

train_data = lgb.Dataset(X_train, label=Y_train)
clf = lgb.train(params, train_data, num_boost_round=3)


y_train = clf.predict(X_train)
y_pred = clf.predict(X_test)

roc_auc2 = roc_auc_score(Y_train, y_train)
roc_auc = roc_auc_score(Y_test, y_pred)
st.write(f"ROC AUC: {roc_auc2:.4f}")
st.write(f"ROC AUC: {roc_auc:.4f}")
st.success("This seems much much better than any other model we tried")

st.header("Try Your Own Data!")

pregnancies = st.number_input('Pregnancies', min_value=0, value=1)
glucose = st.number_input('Glucose', min_value=0, value=100)
blood_pressure = st.number_input('Blood Pressure', min_value=0, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, value=20)
insulin = st.number_input('Insulin', min_value=0, value=80)
bmi = st.number_input('BMI', min_value=0.0, value=25.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, value=0.5)
age = st.number_input('Age', min_value=0, value=30)

if st.button("Predict Diabetes Risk"):
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [dpf],
        'Age': [age]
    })

    prediction = clf.predict(input_data)[0]

    st.subheader("Prediction Result")
    st.write(f"Predicted Probability of Having Diabetes: **{prediction:.2f}**")

    if prediction > 0.5:
        st.error("High Risk of Diabetes")
    else:
        st.success("Low Risk of Diabetes")
        
st.subheader("Final notes")
st.write("""
Throughout testing different models, we consistently encountered the same issue — overfitting. 
One reason for this could be the dataset itself. 
The data we used to train all our models was relatively small, with only around 767 entries. 
Additionally, there were several issues within the dataset that may have introduced bias, such as replacing zero values with averages. 
In the future, we could try finding a larger and cleaner dataset to train on and compare results.
""")
st.write("""
Lastly, we would like to give a huge thanks to CDS (Cornell Data Science) and Sri Kundurthy for providing guidance and support throughout the entire project.
""")
st.text("by: Naijei Jiang(NJ277) & Clarissa McGhee(CM2259)")