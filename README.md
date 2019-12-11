# CoupledCF

This is our implementation for the paper:
Zhang Q, Cao L, Zhu C, et al. CoupledCF: Learning Explicit and Implicit User-item Couplings in Recommendation for Deep Collaborative Filtering[C]//IJCAI. 2018: 3662-3668.

## Please cite our IJCAI'18 paper if you use our codes. Thanks!

# The structure of code is as below:

* mainMovieUserCnn.py: lCoupledCF, gCoupledCF and CoupledCF for MovieLens.
* mainTafengUserCnn.py: lCoupledCF, gCoupledCF and CoupledCF for Tafeng.
* mainMovieUserCnn_only_deepCF.py: DeepCF for MovieLens.
* mainTafengUserCnn_only_deepCF.py: DeepCF for Tafeng.
* LoadMovieDataCnn.py: load data for Movielens.
* LoadTafengDataCnn.py: load data for Tafeng.
* evaluateMovieCnn.py: evaluate lCoupledCF, gCoupledCF and CoupledCF for MovieLens.
* evaluateTafengCnn.py: evaluate lCoupledCF, gCoupledCF and CoupledCF for Tafeng.
* evaluateMovieCnn_only_deepCF.py: evaluate for DeepCF of Movielens.
* evaluateTafengCnn_only_deepCF.py: evaluate for DeepCF of Tafeng.
* ml-1m: Movielens dataset.
* tafeng: Tafeng dataset.
* Pretrain: Predicted results.

The code is implemented in Python based on Keras 2.0.8. It requires pydot and scikit-learn packages to run the code.

Four variants CoupledCF models: DeepCF, lCoupledCF, gCoupledCF and CoupledCF. 
change the value of 'theModel' with the key in 'model_dict' to load different models
of lCoupledCF, gCoupledCF and CoupledCF. For DeepCF, run mainMovieUserCnn_only_deepCF.py
and mainTafengUserCnn_only_deepCF.py.
```
# load model
model_dict={
    "lCoupledCF":get_lCoupledCF_model,
    "gCoupledCF":get_gCoupledCF_model,
    "CoupledCF":get_CoupledCF_model
}

def get_model(theModel,num_users, num_items):
    return model_dict.get(theModel)(num_users, num_items)
    
theModel="lCoupledCF"
    model=get_model(theModel,num_users, num_items)
