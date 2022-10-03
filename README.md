# osman

**osman**: OverSampling by Deep Generative Models 


A pip package which oversamples class imbalanced binary data by Deep Generative Models. 

This package offers two APIs. 

1) Variational Auto Encoders 

2) WGAN-GP


The APIs are very simple to use. One example as follows:

```python

from osman.oversampler import VAEify 

X_vae,y_vae=VAEify(X_train,y_train)

logreg.fit(X_vae,y_vae)

y_pred_vae=logreg.predict(X_test)

score_vae=roc_auc_score(y_pred_vae,y_test)


```
