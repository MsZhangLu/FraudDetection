# Fraud Detection
Nov. 2018

Lu Zhang

## Goal: 
The goal of this challenge is to build a machine learning model that predicts the probability that the first transaction of new user is fraudulent.

## <a id='summary' style='color:black'>Outline: </a>
1. Data Preprocessing
2. Exploratory Data Analysis & Feature Engineering
3. Model Training and Selection
4. Hyperparameter Tuning
5. Findings & Suggestion

### [1. Data Preprocessing](#data_processing)

For this analysis, I will use `pandas` for data manipulation, `matplotlib` for plotting, and `sklearn` for machine learning. 

**1.1 Data loading and browsing**

**1.2 Data preparation**: Join dataframes

**1.3 Data cleansing**

**1.4 Data transformation**: Transform time-series variable

### [2. Exploratory Data Analysis and Feature Engineering](#eda)
* I performed EDA with target, checked relationship between variables and response, and generated features as necessary.

**2.1 Transaction Attributes:**
* `age`, `country`: Categorical features with too many levels.I binned the features down to reduce model complexity

* `purchase_value`: Fraudster are tend to place many orders with identical `purchase_value` . Feature `order_cnt` was generated to monitor how many orders with same value a customer place.

**2.2 Transaction Digital Foot Prints:**
* `source`: Fraud transactions happend more from direct visiting.
* `browser`: Fraudsters use chrome more often than other browsers.

**2.3 Times:**
* In order to take advantage of time series variables, we could create `day_of_week`, `hour_of_day` features for signup and purchase respectively. 
* Fraudsters are more likely to place order immediately after signing up without browsing content of webpage. So we could generate feature `signup_purchase_delta` to cap this information.

**2.4 Transaction Frequencies:**
* Frausters may use same `device_id` or same `ip_address` for multiple transactions. More than half of users who signed up with same device more than once are fraudulent. So we could generate features `signup_anomaly` and `ip_anomaly` to monitor users' signup and purchase behavior.
* We could also generate `{count}_signup_last_{time_window}_by_{device,ip}` to measure user transactions' frequency. 

### [3. Model Building](#build)
First, we'll split variables into features (X) and target (y), and split data into a training(70%) and test(30%) set. 

**3.1 Metrics for Model Comparison**
* precision, recall, f1-score: Since our data is highly imbalanced, we should consider both precision and racall, rather than just accuracy.

* ROC plot and AUC

* Profit curve

**3.2 Model building preparation**

**3.3 Logistic regression (with regularization)**

**3.4 Random forest**

**3.5 Gradient boosting tree**

* Model comparision:

Model|Accuracy|Precision|Recall|F1-score|AUC
-----|--------|---------|------|--------|----
Logistic Regression|0.9570|0.9988|0.5252|0.6884|0.8260
Random Forest|0.9570|0.9988|0.5252|0.6884|0.8283
Gradient Boosting Tree|0.9569|0.9942|0.5264|0.6887|0.8293

* **Without any model tuning, gradient boosting seems to have the best performance** based on the metrics: highest recall, ROC plot with highest AUC. 

**3.6 Down-sampling:**

* All the 3 models' performance have very high precision and poor recall. This is because the data set is imbalanced. 
* Models built on imbalanced data tend to show a bias to the majority class, treating the minority class as noise. 
* Moreover, model evaluation metrics have problems, too. When accuracy seems very high, it may just ignore the minority class. So the classifiers are unreliable.
* Methods dealing with imbalanced data including, down-sampling, and over-sampling. Here I applied down-sampling, and got model performance below.

Model|Accuracy|Precision|Recall|F1-score|AUC
-----|--------|---------|------|--------|----
Random Forest|0.8179|0.9312|0.6829|0.7879|0.8349
Gradient Boosting Tree|0.8177|0.9333|0.6808|0.7873|0.8311

Sampling does help to control precision, accuracy, and boost recall on imbalanced dataset.

### [4. Hyperparameter Tuning](#tune)
I adopted `GridSearchCV` in scikit-learn for hyperparameter tuning. It runs through each combination of search parameters, and compares them based on scoring method `roc_auc_score`. Model with best performance:

Model|Accuracy|Precision|Recall|F1-score|AUC
-----|--------|---------|------|--------|----
Random Forest|0.8185|0.9331|0.6826|0.7886|0.8359
Gradient Boosting Tree|0.8188|0.9374|0.6796|0.7880|0.8353

Random forest classifier has a better performance after tuning. 

Save the **random forest classifier** with parameter below as **final model** to disk:

> RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=20, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=-1,
            oob_score=False, random_state=None, verbose=0,
            warm_start=False)
            
### [5. Findings & Suggestion](#findings)
**5.1 Findings - Fraud chracteristics**
* Fraudsters signup on same device for multiple times.
* Fraudsters take care about their ip address. (They may use VPN.) Signing up multiple times at same ip address may be just happened to normal users.
* Fraudsters make purchase immediately after signing up.
* Fraudsters place many orders with identical purchase value.
* Fraudsters more likely to visit the website directly, instead of searching or converting by ads.
* Fraudsters are more likely to use Chrome than other browsers.

**5.2 Suggestions**
* Limit products' purchasing amount per day. 
* Ask customer fillin CAPTCHA when he/she purchase for the first time. 
* Limit signup times on each device, in case frausters signup multiple times in a short time.
* Increase chance of showing CAPTCHA for user validation, if the user comes to visit the website directly, or visits by Chrome. 

### [6. Evaluation](#eval)
Evaluate the model by unseen data by feeding in test_data.csv. Run the following steps. We can get the predictions' accuracy at the end.
