"""
NumPy House Price Regression

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - impute_nan_with_mean
def impute_nan_with_mean(X):
    """Replace every NaN in X with that column's nan-aware mean (all-NaN cols -> 0).

    Args:
        X: (N, F) array-like of floats, may contain NaN.

    Returns:
        (N, F) float ndarray with no NaNs.
    """
    # TODO: Replace every NaN with that column's nan-aware mean...
    n,f=X.shape
    mu=np.zeros(f)
    mu=np.nanmean(X,axis=0)
    mu_d=np.nan_to_num(mu,nan=0)
    mask=np.isnan(X)
    X_f=np.where(mask, mu_d, X)
    return X_f
    
    pass

# Step 2 - compute_iqr_bounds
def compute_iqr_bounds(X, k=1.5):
    # TODO: Compute per-column lower/upper clip bounds using the IQR rule.
    Q1,Q3=np.percentile(X,[25,75],axis=0)
    IQR=Q3-Q1
    lower=Q1-k*IQR
    upper=Q3+k*IQR
    return np.vstack((lower,upper))
    pass

# Step 3 - clip_columns
def clip_columns(X, lower, upper):
    # TODO: Clip every entry of a feature matrix to per-column lower/upper bounds.
    return np.minimum(np.maximum(X,lower),upper)
    pass

# Step 4 - make_ratio_feature
def make_ratio_feature(numerator, denominator, eps=1e-8):
    # TODO: Form a derived ratio feature from two 1-D arrays with safe division.
    X=numerator/(denominator+eps)
    return X
    pass

# Step 5 - append_column
def append_column(X, col):
    # TODO: Horizontally append one 1-D feature column onto a design matrix.
    return np.column_stack((X,col))
    pass

# Step 6 - one_hot_encode
def one_hot_encode(labels):
    # TODO: Convert a 1-D array of categorical labels into a dense binary one-hot matrix.
    X=np.unique(labels)
    Y=(labels[:,None]==X[None,:])
    return Y.astype(float)
    pass

# Step 7 - fit_standardizer
def fit_standardizer(X):
    # TODO: Compute per-column mean and std used to standardize features...
    M=np.mean(X,axis=0)
    S=np.std(X,axis=0)
    S[S==0]=1.0
    return (M,S)
    pass

# Step 8 - apply_standardizer
def apply_standardizer(X, mean, std):
    # TODO: Return the scaled matrix (X - mean) / std via broadcasting.
    return (X-mean)/std
    pass

# Step 9 - add_bias_column
def add_bias_column(X):
    # TODO: Prepend a column of ones to a 2-D feature matrix X...
    N,F=X.shape
    b=np.ones((N,1))
    return np.hstack((b,X))
    pass

# Step 10 - make_shuffled_indices
def make_shuffled_indices(n_samples, seed):
    # TODO: Create a reproducibly shuffled permutation of row indices.
    rng=np.random.RandomState(seed)
    indices=np.arange(n_samples)
    return rng.permutation(indices)
    pass

# Step 11 - partition_indices
def partition_indices(indices, train_ratio, val_ratio):
    # TODO: Split a shuffled index array into train, validation, and test index arrays.
    N=len(indices)
    n1=int(N*train_ratio)
    n2=int(val_ratio*N)
    n3=N-n1-n2
    train_idx=indices[:n1]
    val_idx=indices[n1:n2+n1]
    test_idx=indices[n2+n1:]
    return (train_idx,val_idx,test_idx)
    pass

# Step 12 - subset_xy
def subset_xy(X, y, indices):
    # TODO: Select the rows of X and y at the given indices.
    return (X[indices],y[indices])
    pass

# Step 13 - ols_fit
def ols_fit(X, y):
    # TODO: return the ordinary-least-squares weight vector for a linear model.
    theta,residuals,rank,s=np.linalg.lstsq(X,y,rcond=None)
    return theta
    pass

# Step 14 - ols_predict
def ols_predict(X, theta):
    # TODO: Predict continuous targets with a fitted linear model.
    return X @ theta
    pass

# Step 15 - mean_absolute_error
def mean_absolute_error(y_true, y_pred):
    # TODO: return the mean absolute error between targets and predictions
    return np.mean(np.abs(y_pred-y_true))
    pass

# Step 16 - root_mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    """Compute root mean squared error between targets and predictions.

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (N,).
        y_pred (np.ndarray): Predicted targets, shape (N,).

    Returns:
        float: RMSE value.
    """
    # TODO: return the root mean squared error as a Python float
    y=y_pred-y_true
    y=y*y
    return np.sqrt(np.mean(y))
    pass

# Step 17 - r_squared
def r_squared(y_true, y_pred):
    # TODO: Compute R^2 = 1 - SS_res/SS_tot (return 0.0 if SS_tot is 0)...
    m=np.mean(y_true)
    SSR=np.sum((y_pred-y_true)*(y_pred-y_true))
    SST=np.sum((y_true-m)*(y_true-m))
    if SST==0:
        return 0
    return 1-(SSR/SST)
    pass

# Step 18 - residual_summary
def residual_summary(y_true, y_pred):
    # TODO: Return a compact dict summarizing prediction residuals...
    r=y_true-y_pred
    m=np.mean(r)
    s=np.std(r)
    med=np.median(np.abs(r))
    return {'mean':m,'std':s,'median_abs':med}
    pass

# Step 19 - prepare_cleaned_features
def prepare_cleaned_features(X, iqr_k=1.5):
    """Impute NaNs then IQR-clip columns to produce a clean numeric matrix.

    Args:
        X: (N, F) array-like of floats, may contain NaN.
        iqr_k: IQR multiplier passed to compute_iqr_bounds (default 1.5).

    Returns:
        (N, F) float ndarray with no NaNs, columns clipped to IQR bounds.
    """
    # TODO: Produce a clean numeric matrix via impute then IQR clip
    X1=impute_nan_with_mean(X)
    X2=compute_iqr_bounds(X1,iqr_k)
    lower=X2[0]
    upper=X2[1]
    return clip_columns(X1,lower,upper)
    pass

# Step 20 - assemble_feature_matrix
import numpy as np
def assemble_feature_matrix(X_num, ratio_num_idx, ratio_den_idx, cat_labels=None):
    # TODO: build an extended feature matrix by appending a derived ratio...
    X1=make_ratio_feature(X_num[:,ratio_num_idx],X_num[:,ratio_den_idx],eps=1e-8)
    X2=append_column(X_num,X1)
    if cat_labels is not None:
        X3=one_hot_encode(cat_labels)
        X2=append_column(X2,X3)
    return X2
    pass

# Step 21 - make_train_val_test
def make_train_val_test(X, y, train_ratio, val_ratio, seed):
    # TODO: Shuffle and materialize train/validation/test matrices from X and y...
    n=X.shape[0]
    s_indices=make_shuffled_indices(n,seed)
    X1=partition_indices(s_indices,train_ratio,val_ratio)
    s={'X_train':X[X1[0]],'y_train':y[X1[0]],'X_val':X[X1[1]],'y_val':y[X1[1]],'X_test':X[X1[2]],'y_test':y[X1[2]]}
    return s
    pass

# Step 22 - standardize_and_add_bias
def standardize_and_add_bias(splits):
    # TODO: Fit standardizer on train, transform all splits, prepend bias...
    X1_data=fit_standardizer(splits['X_train'])
    mean=X1_data[0]
    std=X1_data[1]
    answer={'X_train':None,'y_train':None,'X_val':None,'y_val':None,'X_test':None,'y_test':None}
    answer['X_val']=apply_standardizer(splits['X_val'],mean,std)
    answer['X_test']=apply_standardizer(splits['X_test'],mean,std)
    answer['X_train']=apply_standardizer(splits['X_train'],mean,std)
    answer['y_train']=splits['y_train']
    answer['y_val']=splits['y_val']
    answer['y_test']=splits['y_test']
    for k,v in answer.items():
        if k=='X_train' or k=='X_val' or k=='X_test':
         answer[k]=add_bias_column(v)
    return (answer,mean,std)
    pass

# Step 23 - evaluate_predictions
def evaluate_predictions(y_true, y_pred):
    
    # TODO: Bundle MAE, RMSE, R^2, and residual summary into one metrics dict.
    x1=mean_absolute_error(y_true,y_pred)
    x2=root_mean_squared_error(y_true,y_pred)
    x3=r_squared(y_true,y_pred)
    x4=residual_summary(y_true,y_pred)
    ans={'mae':x1,'rmse':x2,'r2':x3,'residual_summary':x4}
    return ans
    pass

# Step 24 - house_price_pipeline
def house_price_pipeline(X, y, ratio_num_idx, ratio_den_idx, cat_labels=None, train_ratio=0.7, val_ratio=0.15, seed=42, iqr_k=1.5):
    # TODO: Run full clean->featurize->split->standardize->OLS->evaluate pipeline...
    X_clean=prepare_cleaned_features(X,iqr_k)
    X_feat=assemble_feature_matrix(X_clean,ratio_num_idx,ratio_den_idx,cat_labels)
    splits=make_train_val_test(X_feat,y,train_ratio,val_ratio,seed)
    std_splits,mean,std=standardize_and_add_bias(splits)
    theta=ols_fit(std_splits['X_train'],std_splits['y_train'])
    y_hat_val=ols_predict(std_splits['X_val'],theta)
    y_hat_test=ols_predict(std_splits['X_test'],theta)
    val_metrics=evaluate_predictions(std_splits['y_val'],y_hat_val)
    test_metrics=evaluate_predictions(std_splits['y_test'],y_hat_test)
    return {'theta':theta,'y_test':std_splits['y_test'],'y_test_pred':y_hat_test,'test_metrics':test_metrics,'val_metrics':val_metrics}

    pass

