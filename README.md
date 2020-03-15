# AIAP Technical Assessment

Design and create a simple machine learning pipeline that will ingest/process the entailed
dataset and feed it into the machine learning algorithm(s) of your choice, returning metric outputs. 

## Explanation of mlp.py
This is an end to end machine learning pipeline that considers 5 regression models:
1. Linear Regression 
2. Support Vector Regression
3. Decision Tree Regression
4. Random Forest Regression
5. XGBoost Regression

The reason for choosing the 5 models is because the dependent variable is continuous, 
and the distribution of the dependent variable is non-Gaussian.

Data was scaled using MinMaxScaler as StandardScaler would remove the unique feature of the dataset,
which is that it has a long right tail.

### Prerequisites

You will need to install python 3.6.7, and install all packages in their respective versions by running this in the command line

```
pip3 install -r requirements.txt
```

### Running the file

Execute the run.sh file on your terminal, and key in 2 optional arguments.
1st optional argument: Model Evaluation Criteria
* 1 --> RMSE
* 2 --> R2 value
* 3 --> Median Absolute Error
2nd optional argument: Number of folds in cross validation scoring. Value has to be greater than 1.
```
###EXAMPLE###
./run.sh 1 2
```

### Sample Output
```
In 3 rounds, LR model got average neg_root_mean_squared_error score of 0.12449529006115519 with standard deviation of 0.0011329188592213036.
In 3 rounds, SVR model got average neg_root_mean_squared_error score of 0.09488094862013761 with standard deviation of 0.0005461950805368554.
In 3 rounds, DTR model got average neg_root_mean_squared_error score of 0.11328900303052121 with standard deviation of 0.0010234489626689428.
In 3 rounds, RFR model got average neg_root_mean_squared_error score of 0.08710899447489406 with standard deviation of 0.0008629366494751925.
In 3 rounds, XGB model got average neg_root_mean_squared_error score of 0.08701051386807708 with standard deviation of 0.00040105972715497864.
----------------------
    The best model was XGBRegressor(alpha=10, base_score=None, booster=None, colsample_bylevel=None,
             colsample_bynode=None, colsample_bytree=0.3, gamma=None,
             gpu_id=None, importance_type='gain', interaction_constraints=None,
             learning_rate=0.1, max_delta_step=None, max_depth=5,
             min_child_weight=None, missing=nan, monotone_constraints=None,
             n_estimators=200, n_jobs=None, num_parallel_tree=None,
             objective='reg:squarederror', random_state=None, reg_alpha=None,
             reg_lambda=None, scale_pos_weight=None, subsample=None,
             tree_method=None, validate_parameters=False, verbosity=None) with neg_root_mean_squared_error score of 0.08701051386807708, and a neg_root_mean_squared_error standard deviation of 0.00040105972715497864.
```
## Acknowledgments

* Everyone on stackoverflow, thank you

