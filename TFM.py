# -----------LIBRERIAS-----------
import time
import csv
import pandas as pd
import xlrd
import nltk
import stanza
#stanza.download('es')
nlp = stanza.Pipeline('es')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer

# -----------FUNCIONES-----------

# Funcion que quita las tildes de las vocales
def quitar_tildes(palabra):
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        ("Á", "A"),
        ("É", "E"),
        ("Í", "I"),
        ("Ó", "O"),
        ("Ú", "U"),
    )
    for a, b in replacements:
        palabra = palabra.replace(a, b)
    return palabra

# Funcion que obtiene los datos de las recetas en una lista de diccionarios
def recetas_cocina():
    recipes = []
    with open('recetas.csv', encoding="utf8") as File:
        reader = csv.DictReader(File, delimiter='|')
        for row in reader:
            if row['Ingredientes'] != '':
                recipes.append(row)
    # Dejamos los ingredientes como una lista
    for i in range(0, len(recipes)):
        dic = recipes[i]
        Ingredientes = dic['Ingredientes']
        Lista = Ingredientes.split(',')
        dic['Ingredientes'] = Lista
        Lista1 = []
        # Tokenizamos y lematizamos las palabras de los ingredientes de la receta
        try:
            for j in range(0, len(Lista)):
                Lista2 = []
                if Lista[j] != '':
                    doc = nlp(Lista[j])
                    for sent in doc.sentences:
                        for word in sent.words:
                            wordlemma = quitar_tildes(word.lemma).lower()
                            Lista2.append(wordlemma)
                    Lista1.append(Lista2)
        except:
            print('Ha ocurrido un error en la receta con Id: ' + str(dic['Id']))
        dic['Ingredientes'] = Lista1  
    return recipes

# Funcion que obtiene la composicion nutricional de los alimentos en una lista de diccionarios
def composicion_nutricional():
    workbook = xlrd.open_workbook('ficha_de_alimentación_transformada.xlsx')
    worksheet = workbook.sheet_by_index(0)
    first_row = [] # The row where we stock the name of the column
    for col in range(worksheet.ncols):
        first_row.append( worksheet.cell_value(0,col) )
    alimentos = []
    for row in range(1, worksheet.nrows):
        elm = {}
        for col in range(worksheet.ncols):
            elm[first_row[col]] = worksheet.cell_value(row,col)
        alimentos.append(elm)
    # Tokenizamos y lematizamos las palabras de los alimentos
    for a in range(0,len(alimentos)):
        Lista1 = []
        doc = nlp(alimentos[a]['Alimento'])
        for sent in doc.sentences:
            for word in sent.words:
                wordlemma = quitar_tildes(word.lemma).lower()
                Lista1.append(wordlemma)
        alimentos[a]['Alimento'] = Lista1
    return alimentos

# Funcion que obtiene las equivalencias sobre las medidas
def equivalencias_medidas():
    equivalencias = []
    with open('equivalencias.csv', encoding="utf8") as File2:
        reader = csv.DictReader(File2, delimiter=',')
        for row in reader:
            num = row['gramos']
            doc = nlp(row['medida'])
            for sent in doc.sentences:
                for word in sent.words:
                    lemma = quitar_tildes(word.lemma).lower()
                    dic = {'medida' : lemma, 'gramos' : num}
                    equivalencias.append(dic)
    return equivalencias

# Funcion que saca el ingrediente
def sacar_ingrediente(ingrediente):
    ingred = 'desconocido' #fijamos esta por si no encuentra y la composicion a 0
    composicion = {'CAL': 0, 'PR': 0, 'GR': 0, 'HC': 0, 'H20': 0, 
    'CEN': 0, 'A': 0, 'B1': 0, 'B2': 0, 'C': 0, 'Niac': 0, 'Na': 0, 'K': 0, 
    'Ca': 0, 'Mg': 0, 'Fe': 0, 'Cu': 0, 'P': 0, 'S': 0, 'Cl': 0, 'Fen': 0, 
    'Ileu': 0, 'Leu': 0, 'Lis': 0, 'Met': 0, 'Tre': 0, 'Tri': 0, 'Val': 0}
    posibilidades = []
    posibilidades_maximas = []
    posibilidad_final = {}
    max_cont = 0
    cont = 0
    # Recorremos cada alimento con su composicion nutricional
    for i in range(0,len(alimentos)):
        cont = 0
        alimento = alimentos[i]
        nombre_alimento = alimento['Alimento']
        for j in range(0,len(nombre_alimento)):
            palabra_alimento = nombre_alimento[j]
            for k in range(0,len(ingrediente)):
                palabra_ingrediente = ingrediente[k]
                if palabra_alimento == palabra_ingrediente:
                    cont = cont + 1
                    alimento['cont'] = cont
                else:
                    alimento['cont'] = cont
        if cont == len(nombre_alimento):
            posibilidades.append(alimento)
            if max_cont < cont:
                max_cont = cont
    for l in range(0, len(posibilidades)):
        posibilidad = posibilidades[l]
        if posibilidad['cont'] == max_cont:
            posibilidades_maximas.append(posibilidad)
    for m in range(0, len(posibilidades_maximas)):
        posibilidad_max = posibilidades_maximas[m]
        if posibilidad_max['Estado'].lower() in ['crudo', 'cruda']:
            posibilidad_final = posibilidad_max
            break
    if posibilidad_final == {}:
        for n in range(0, len(posibilidades_maximas)):
            posibilidad_max = posibilidades_maximas[n]
            if posibilidad_max['Estado'].lower() in ['fresco', 'fresca']:
                posibilidad_final = posibilidad_max
                break
    if posibilidad_final == {}:
        for o in range(0, len(posibilidades_maximas)):
            posibilidad_max = posibilidades_maximas[o]
            if posibilidad_max['Estado'].lower() == '':
                posibilidad_final = posibilidad_max
                break
    if posibilidad_final == {} and len(posibilidades_maximas)>0:
        posibilidad_final = posibilidades_maximas[0]
    if len(posibilidades)>0:
        for p in composicion:
            composicion[p] = posibilidad_final[p]
    return posibilidad_final, composicion

# Funcion que saca la cantidad
def sacar_cantidad(ingrediente): 
    # Recorremos cada palabra del ingrediente
    for k in range(0,len(ingrediente)):
        palabra = ingrediente[k]
        cantidad = 1 # fijamos esto por si no encuentra
        for k in range(0,len(ingrediente)):
            palabra = ingrediente[k]
            try:
                cantidad = float(palabra)
            except ValueError:
                pass
    return cantidad

# Funcion que saca la equivalencia
def sacar_equivalencia(ingrediente):
    # Recorremos cada palabra del ingrediente
    for k in range(0,len(ingrediente)):
        medida = 'desconocido' #fijamos esta por si no encuentra
        gramos = 100 #fijamos esta por si no encuentra
        palabra = ingrediente[k]
        for l in range(0,len(equivalencias)):
            equivalencia = equivalencias[l]
            if equivalencia['medida'].lower() == palabra.lower():
                medida = equivalencia['medida'].lower()
                gramos = float(equivalencia['gramos'])
                break 
        else:
            continue
        break
    return medida, gramos

# Funcion que calcula la composicion nutricional de las recetas
def composicion_recetas(ingredientes, raciones):
    composicion_receta = {'CAL': 0, 'PR': 0, 'GR': 0, 'HC': 0, 'H20': 0, 'CEN': 0, 
    'A': 0, 'B1': 0, 'B2': 0, 'C': 0, 'Niac': 0, 'Na': 0, 'K': 0, 'Ca': 0, 'Mg': 0, 
    'Fe': 0, 'Cu': 0, 'P': 0, 'S': 0, 'Cl': 0, 'Fen': 0, 'Ileu': 0, 'Leu': 0, 'Lis': 0, 
    'Met': 0, 'Tre': 0, 'Tri': 0, 'Val': 0}
    for i in range(0,len(ingredientes)):
        ingrediente = ingredientes[i]
        ingr, composicion_ingr = sacar_ingrediente(ingrediente)
        cant = sacar_cantidad(ingrediente)
        med, gr = sacar_equivalencia(ingrediente)
        gramos = cant * gr / 100
        for j in composicion_receta:
            composicion_receta[j] = composicion_receta[j] + (composicion_ingr[j] * gramos)
    for k in composicion_receta:
        if raciones == '':
            raciones = 1
        composicion_receta[k] = round((composicion_receta[k] / int(raciones)), 3)
    return composicion_receta

# ------------------------------------------------------------

# ESTRUCTURA PRINCIPAL
# Para ver el tiempo que ha tarda el proceso
print('Iniciando el proceso')
start_time = time.time()
# Lectura de datos
recipes = recetas_cocina()
alimentos = composicion_nutricional()
equivalencias = equivalencias_medidas()
# Cruce de datos
for i in range(0,len(recipes)):
    # Recorremos los ingredientes de la receta
    receta = recipes[i]
    ingredientes = receta['Ingredientes']
    raciones = receta['Num_comensales']
    composicion_receta = composicion_recetas(ingredientes,raciones)
    for j in composicion_receta:
        receta[j] = composicion_receta[j]
#Guarda la información de las recetas en un fichero CSV
keys = recipes[0].keys()
with open("RecetasComposicion.csv", mode='w', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter='|', quotechar='"', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(keys)
    for r in range(0,len(recipes)):
        receta = recipes[r]
        values = receta.values()
        writer.writerow(values)
# Vemos cuanto tiempo ha durado
end_time = time.time()
total_time = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
print ('\nEl proceso ha tardado: ', total_time)



