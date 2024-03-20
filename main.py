import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import dataset
dataset = pd.read_csv('Data.csv')
features = dataset.iloc[:, :-1].values  # index locating ( row & column )
dependent_variable = dataset.iloc[:, -1]
