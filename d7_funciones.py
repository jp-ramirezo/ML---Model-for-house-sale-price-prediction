# Librerías básicas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import missingno as msno

# Métodos asociados a regularización
from sklearn.metrics import classification_report, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# Especificaciones
warnings.filterwarnings(action= 'ignore')
plt.style.use('seaborn-whitegrid')


def grafestad_numbers(dataframe, ajuste):
    '''
    Devuelve el histograma de las frecuencias de atributos de tipo float o int. Adicionalmente, genera dos líneas verticales que indican la media (rojo) y moda (azul) del atributo analizado.
    ----------
    Elementos:
        
    - dataframe: Dataframe a analizar
    - ajuste: (int) valor de separación lateral de los gráficos.
        
    '''
    variable = []
    for col in dataframe:
        if (dataframe[col].dtypes == 'float64') or (dataframe[col].dtypes == 'Int64') or (dataframe[col].dtypes == 'Int32'):
            variable.append(col)
    for i in range (len(variable)):
        plt.subplot(2,len(variable),i+1)
        plt.hist(dataframe[variable[i]].dropna())
        plt.title(f'Frecuencias de {variable[i]}')
        plt.axvline(dataframe[variable[i]].mean(), color= 'tomato')
        plt.axvline(dataframe[variable[i]].median(), color= 'blue')
        plt.xlabel('')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=30)
    return plt.subplots_adjust(right=ajuste, hspace=.5)
    
def grafestad_object(dataframe, ajuste):
    '''
    Devuelve el histograma de las frecuencias de atributos de tipo object.
    Elementos:
        
    - dataframe: Dataframe a analizar
    - ajuste: (int) valor de separación lateral de los gráficos.
        
    '''
    variable = []
    for col in dataframe:
        if dataframe[col].dtypes == 'object':
            variable.append(col)
    for i in range (len(variable)):
        plt.subplot(2,len(variable),i+1)
        sns.countplot(dataframe[variable[i]].dropna())
        plt.title(f'Frecuencias de {variable[i]}')
        plt.xlabel('')
        plt.ylabel('Frecuencia')
        plt.xticks(rotation=90)
    return plt.subplots_adjust(right=ajuste, hspace=.5)

def graficador(dataframe, rows, cols):
    '''
    '''
    for index, (colname, serie) in enumerate (dataframe.iteritems()):
        plt.subplot(rows, cols, index + 1)
        if serie.dtype == 'object':
            sns.countplot(serie.dropna())
            plt.axhline(serie.value_counts().mean(), color='forestgreen', linestyle= '--')
            plt.xticks(rotation=45)
            plt.title(colname)
        else:
            sns.distplot(serie.dropna(), color= 'slategrey')
            plt.axvline(serie.mean(), color= 'forestgreen', linestyle= '--')
            plt.title(colname)
    plt.subplots_adjust(right=2.5, hspace=2)

def desempeño(test, hat):
    print('- Métricas de desempeño:')
    print(f'MSE: {mean_squared_error(test, hat).round(3)}')
    print(f'MAE: {median_absolute_error(test, hat).round(3)}')
    print(f'R2 : {r2_score(test, hat).round(3)}')
    
def plot_importance(fit_model, feat_names):
    """TODO: Docstring for plot_importance.

    :fit_model: TODO
    :: TODO
    :returns: TODO

    """
    tmp_importance = fit_model.feature_importances_
    sort_importance = np.argsort(tmp_importance)[::-1]
    names = [feat_names[i] for i in sort_importance]
    plt.title("Feature importance")
    plt.barh(range(len(feat_names)), tmp_importance[sort_importance])
    plt.yticks(range(len(feat_names)), names, rotation=0)