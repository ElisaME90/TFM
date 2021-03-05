# -----------LIBRERIAS-----------
import numpy as np
import pandas as pd
import time
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# -----------FUNCIONES-----------

# Funcion que lee los datos nutricionales del fichero
def recetas_composicion():
    recipes_data = pd.read_csv('RecetasComposicion.csv', sep="|")
    return recipes_data

# Gráficos de cada componente nutricional por separado
def graficos_composicion(dataset):
    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.suptitle('Composición Básica')
    plt.plot(dataset['CAL'], 'ro')
    #plt.xlabel('CAL')
    plt.title('Calorías')
    plt.subplot(132)
    plt.plot(dataset['PR'], 'ro')
    #plt.xlabel('PR')
    plt.title('Proteínas')
    plt.subplot(133)
    plt.plot(dataset['GR'], 'ro')
    #plt.xlabel('GR')
    plt.title('Grasas')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(131)
    plt.suptitle('Composición Básica')
    plt.plot(dataset['HC'], 'ro')
    plt.title('Hidratos de carbono')
    plt.subplot(132)
    plt.plot(dataset['H20'], 'ro')
    plt.title('Agua')
    plt.subplot(133)
    plt.plot(dataset['CEN'], 'ro')
    plt.title('Cenizas')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(151)
    plt.suptitle('Vitaminas')
    plt.plot(dataset['A'], 'ro')
    plt.title('A')
    plt.subplot(152)
    plt.plot(dataset['B1'], 'ro')
    plt.title('B1')
    plt.subplot(153)
    plt.plot(dataset['B2'], 'ro')
    plt.title('B2')
    plt.subplot(154)
    plt.plot(dataset['C'], 'ro')
    plt.title('C')
    plt.subplot(155)
    plt.plot(dataset['Niac'], 'ro')
    plt.title('Niacina')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.suptitle('Minerales')
    plt.subplot(151)
    plt.plot(dataset['Na'], 'ro')
    plt.title('Sodio')
    plt.subplot(152)
    plt.plot(dataset['K'], 'ro')
    plt.title('Potasio')
    plt.subplot(153)
    plt.plot(dataset['Ca'], 'ro')
    plt.title('Calcio')
    plt.subplot(154)
    plt.plot(dataset['Mg'], 'ro')
    plt.title('Magnesio')
    plt.subplot(155)
    plt.plot(dataset['Fe'], 'ro')
    plt.title('Hierro')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.suptitle('Minerales')
    plt.subplot(141)
    plt.plot(dataset['Cu'], 'ro')
    plt.title('Cobre')
    plt.subplot(142)
    plt.plot(dataset['P'], 'ro')
    plt.title('Fosforo')
    plt.subplot(143)
    plt.plot(dataset['S'], 'ro')
    plt.title('Azufre')
    plt.subplot(144)
    plt.plot(dataset['Cl'], 'ro')
    plt.title('Cloro')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(141)
    plt.suptitle('Aminoácidos')
    plt.plot(dataset['Fen'], 'ro')
    plt.title('Fenilalanina')
    plt.subplot(142)
    plt.plot(dataset['Ileu'], 'ro')
    plt.title('Isoleucina')
    plt.subplot(143)
    plt.plot(dataset['Leu'], 'ro')
    plt.title('Leucina')
    plt.subplot(144)
    plt.plot(dataset['Lis'], 'ro')
    plt.title('Lisina')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.suptitle('Aminoácidos')
    plt.subplot(141)
    plt.plot(dataset['Met'], 'ro')
    plt.title('Metionina')
    plt.subplot(142)
    plt.plot(dataset['Tre'], 'ro')
    plt.title('Treonina')
    plt.subplot(143)
    plt.plot(dataset['Tri'], 'ro')
    plt.title('Triptófano')
    plt.subplot(144)
    plt.plot(dataset['Val'], 'ro')
    plt.title('Valina')
    plt.show()

def escalar_datos(dataset):
    scaler = MinMaxScaler()
    transformed_recipes_data = scaler.fit_transform(dataset.iloc[:, 14:])
    transformed_recipes_data = pd.DataFrame(transformed_recipes_data)
    #print(transformed_recipes_data)
    transformed_recipes_data['Id'] = dataset['Id']
    transformed_recipes_data['Categoria'] = dataset['Categoria']
    transformed_recipes_data['Nombre'] = dataset['Nombre']
    transformed_recipes_data['Valoracion'] = dataset['Valoracion']
    transformed_recipes_data['Dificultad'] = dataset['Dificultad']
    transformed_recipes_data['Num_comensales'] = dataset['Num_comensales']
    transformed_recipes_data['Tiempo'] = dataset['Tiempo']
    transformed_recipes_data['Tipo'] = dataset['Tipo']
    transformed_recipes_data['Link_receta'] = dataset['Link_receta']
    transformed_recipes_data['Num_comentarios'] = dataset['Num_comentarios']
    transformed_recipes_data['Num_reviews'] = dataset['Num_reviews']
    transformed_recipes_data['Fecha_modificacion'] = dataset['Fecha_modificacion']
    transformed_recipes_data['Ingredientes'] = dataset['Ingredientes']
    transformed_recipes_data['Pasos_elaboracion'] = dataset['Pasos_elaboracion']
    #print(transformed_recipes_data)
    transformed_recipes_data = transformed_recipes_data[['Id', 'Categoria', 'Nombre',
     'Valoracion', 'Dificultad', 'Num_comensales', 'Tiempo', 'Tipo', 'Link_receta', 
     'Num_comentarios', 'Num_reviews', 'Fecha_modificacion', 'Ingredientes', 
     'Pasos_elaboracion', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
     17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]]
    transformed_recipes_data.columns = dataset.columns
    return transformed_recipes_data

def kmeans_CAL_PR(dataset):
    # Iniciamos el objeto KMeans especificando el nº de clústeres deseados
    #x = recipes_data.copy()
    x = dataset.iloc[:, 14:16].copy()
    print(x)
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(x)
    clusters = x.copy()
    clusters['cluster_pred'] = kmeans.fit_predict(x)
    plt.figure(figsize=(14, 7))
    plt.scatter(clusters['CAL'], clusters['PR'], c= clusters['cluster_pred'], cmap='rainbow')
    plt.xlabel('CAL')
    plt.ylabel('PR')
    plt.suptitle('Kmeans, k=5')
    plt.show()

def Busqueda_k(dataset,grupo, a, b, x, y):
    # Iniciamos el objeto KMeans especificando el nº de clústeres deseados
    rango = range(x, y)
    x = dataset.iloc[:, a:b].copy()
    print(x)
    kmeans = [KMeans(n_clusters=k) for k in rango]
    kmeans
    score = [kmeans[i].fit(x).score(x) for i in range(len(kmeans))]
    score
    plt.plot(rango,score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Curva de codo: ' + grupo)
    plt.show()

def clustering_kmeans(dataset,grupo, a, b, k):
    x = dataset.iloc[:, a:b].copy()
    kmeans = KMeans(n_clusters=k).fit(x)
    x['Clust_' + grupo[:3]] = kmeans.predict(x)
    print(x)
    centroids = kmeans.cluster_centers_
    print('Grupo : ' + grupo)
    print('Centroides:')
    print(centroids)
    print()
    return x

# ------------------------------------------------------------

# ESTRUCTURA PRINCIPAL
# Para ver el tiempo que ha tarda el proceso
print('Iniciando el proceso')
start_time = time.time()

# Lectura de datos
recipes_data = recetas_composicion()
print(recipes_data)


# Mediante estas líneas se han comprobado los ID de los valores atípicos
"""
recipes = recetas()
print(recipes.head())
for i in recipes.index:
    if recipes['Fen'][i] > 1000000:
        print(recipes['Id'][i])
"""
# Hacemos gráficos de cada uno de los componentes para su análisis
graficos_composicion(recipes_data)

# Escalamos las variables nutricionales
transformed_recipes_data = escalar_datos(recipes_data)
# Hacemos gráficos de cada uno de los componentes, ahora escalados
graficos_composicion(transformed_recipes_data)

# Comprobación de K-means solo para CAL y PR sin escalado
kmeans_CAL_PR(recipes_data)
# Comprobación de K-means solo para CAL y PR con escalado
kmeans_CAL_PR(transformed_recipes_data)

# Buscamos un numero k para cada grupo 
Busqueda_k(transformed_recipes_data, 'Básicos', 14, 20, 1, 20)
Busqueda_k(transformed_recipes_data, 'Vitaminas', 20, 25, 1, 20)
Busqueda_k(transformed_recipes_data, 'Minerales', 25, 34, 1, 20)
Busqueda_k(transformed_recipes_data, 'Aminoácidos', 34, 42, 1, 20)

# Aplicamos el algoritmo K-means a los datos por grupos
transformed_recipes_data_1 = clustering_kmeans(transformed_recipes_data, 'Básicos', 14, 20, 6)
transformed_recipes_data_2 = clustering_kmeans(transformed_recipes_data, 'Vitaminas', 20, 25, 7)
transformed_recipes_data_3 = clustering_kmeans(transformed_recipes_data, 'Minerales', 25, 34, 3)
transformed_recipes_data_4 = clustering_kmeans(transformed_recipes_data, 'Aminoácidos', 34, 42, 3)
transformed_recipes_data['Clust_Bás'] = transformed_recipes_data_1['Clust_Bás']
transformed_recipes_data['Clust_Vit'] = transformed_recipes_data_2['Clust_Vit']
transformed_recipes_data['Clust_Min'] = transformed_recipes_data_3['Clust_Min']
transformed_recipes_data['Clust_Ami'] = transformed_recipes_data_4['Clust_Ami']
print(transformed_recipes_data)
#transformed_recipes_data.to_csv('RecetasClustering.csv', sep="|")

# Vemos cuanto tiempo ha durado
end_time = time.time()
total_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
print ('\nEl proceso ha tardado: ', total_time)

#--------------------------------------------
