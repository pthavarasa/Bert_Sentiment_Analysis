# -*- coding: utf-8 -*-

import linear_regression_with_sql

phrase = input("Enter a sentence: ")
ressenti = input("enter a feeling")
predict_setiment = linear_regression_with_sql.LR_model(phrase,ressenti)

predict_setiment = 'positif' if predict_setiment == 1 else 'negatif'
print('La phrase a été prédite comme : ',predict_setiment)