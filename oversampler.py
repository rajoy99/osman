from helpers import get_cat_dims 

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from WGAN_model import WGANGP
from VAE_model_noCat import VAE_model

def WGANify(X_train,y_train,num_cols=[]):

    if not num_cols:
        num_cols=X_train.columns
    cat_cols=[]

    num_prep = make_pipeline(SimpleImputer(strategy='mean'),
                         MinMaxScaler())
    cat_prep = make_pipeline(SimpleImputer(strategy='most_frequent'),
                            OneHotEncoder(handle_unknown='ignore', sparse=False))

    prep = ColumnTransformer([
        ('num', num_prep, num_cols)],
        remainder='drop')
    X_train_trans = prep.fit_transform(X_train)

    cat_dims=[(X_train[col].nunique()) for col in cat_cols]

    gan = WGANGP(write_to_disk=False, # whether to create an output folder. Plotting will be surpressed if flase
            compute_metrics_every=125000, print_every=250000, plot_every=1000000,
            num_cols = num_cols,
            # pass the one hot encoder to the GAN to enable count plots of categorical variables
            # pass column names to enable
            cat_cols=cat_cols,
            use_aux_classifier_loss=True,
            d_updates_per_g=3, gp_weight=15)

    gan.fit(X_train_trans, y=y_train.values, 
            condition=False,
            epochs=30,  
            batch_size=64,
            netG_kwargs = {'hidden_layer_sizes': (128,64), 
                            'n_cross_layers': 1,
                            'num_activation': 'none',
                            'noise_dim': 30, 
                            'normal_noise': False,
                            'activation':  'leaky_relu',
                            'use_num_hidden_layer': True,
                            'layer_norm':False,},
            netD_kwargs = {'hidden_layer_sizes': (128,64,32),
                            'n_cross_layers': 2,
                            'activation':  'leaky_relu',
                            'sigmoid_activation': False,
                            'noisy_num_cols': True,
                            'layer_norm':True,}
        )

    X_res, y_res = gan.resample(X_train_trans, y=y_train)

    return X_res,y_res

def VAEify(X_train,y_train,num_cols=[]): 
    

    X_res,y_res=VAE_model(X_train,y_train,num_cols)

    return X_res,y_res
