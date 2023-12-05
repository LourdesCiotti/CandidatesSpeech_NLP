# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:00:37 2023

@author: Lufly
"""

"""
Massa
"""
import nltk
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import spacy
import string
import re
from transformers import pipeline
from collections import Counter
import matplotlib.pyplot as plt


# Cargar el modelo en español
nlp = spacy.load('es_core_news_sm')
# Descargar los recursos necesarios de nltk
nltk.download('stopwords')
nltk.download('punkt')

# Ruta al archivo de texto
archivo = r'C:\Users\Lufly\Desktop\Proyectos\milei.txt'
# Leer el contenido del archivo
with open(archivo, 'r', encoding='utf-8') as file:
    texto = file.read()

# Eliminar números y signos de puntuación del texto
translator = str.maketrans('', '', string.digits + string.punctuation)
texto = texto.translate(translator)
# Tokenización: Divide el texto en palabras
words = word_tokenize(texto)
# Convertir todo el texto a minúscula
texto = texto.lower()

# Procesar el texto con spaCy
doc = nlp(texto)
# Identificar y mostrar artículos (DET) y preposiciones (ADP) antes de eliminarlos
palabras_a_eliminar = [token.text for token in doc if (token.pos_ in ['ADP', 'DET'] or token.text.lower() in ['música','es','se','también', 'nos', 'tiene','como', 'cuando', 'eso', 'y', 'de', 'porque', 'que', 'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'lo', 'le', 'uno', 'me',  'sé', 'va', 'además', 'no','son', 'vez', 'ibamos', 'les']) and token.text.lower() != 'sin' and token.text.lower() != 'contra']
# Eliminar las palabras definidas en palabras_a_eliminar del texto
texto_filtrado = ' '.join([word.lower() for word in words if word.lower() not in palabras_a_eliminar])
# Utilizar una expresión regular para eliminar letras sueltas
texto_limpio = re.sub(r'\s+[A-Za-z]\s+', ' ', texto_filtrado)


# Realizar el análisis de sentimientos en el texto preprocesado
sentimientos = TextBlob(texto_limpio)
# Obtener el puntaje de polaridad (positivo/negativo)
polaridad = sentimientos.sentiment.polarity
# Clasificar el sentimiento
if polaridad > 0:
    sentimiento = "Positivo"
elif polaridad < 0:
    sentimiento = "Negativo"
else:
    sentimiento = "Neutro"
# Imprimir los resultados
print("Análisis de Sentimientos del discurso en '{}'".format(archivo))
print("Texto Preprocesado:", texto_limpio)
print("Polaridad del Sentimiento:", polaridad)
print("Clasificación del Sentimiento:", sentimiento)

# SON MUY PESADOS PARA DESCARGAR CON MI BANDA Y COMPUTADORA
# Realizar la clasificación de temas
#clasificador_temas = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
#temas = clasificador_temas(texto_preprocesado, candidate_labels=["Política", "Economía", "Cultura", "Deportes"])
# Resultados de la clasificación de temas
#tema_principal = temas['labels'][0]
#confianza = temas['scores'][0]

# Crear un clasificador de emociones
# clasificador_emociones = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
# # Realizar la detección de emociones
# resultados = clasificador_emociones(texto)
# # Interpretar los resultados
# categorias_emociones = resultados[0]['labels']
# emocion_predominante = categorias_emociones[0]
# # Imprimir los resultados
# print("Texto: ", texto)
# print("Emoción predominante:", emocion_predominante)

# Contar la frecuencia de cada palabra en el texto
frecuencia_palabras = Counter(texto_limpio.split())

# Obtener las palabras más comunes
num_palabras_mostrar = 20  # Puedes ajustar este valor según tus necesidades
palabras_comunes = frecuencia_palabras.most_common(num_palabras_mostrar)

# Extraer palabras y frecuencias para graficar
palabras, frecuencias = zip(*palabras_comunes)

# Crear un gráfico de línea horizontal de las palabras más utilizadas con cuadrícula
plt.figure(figsize=(12, 6))
plt.plot(palabras, frecuencias, marker='o', linestyle='-', color='b', markersize=8)
plt.xlabel('Palabras')
plt.ylabel('Frecuencia')
plt.title('Palabras Más Utilizadas en el discurso')
plt.xticks(rotation=45)
plt.grid(True)  # Agregar un fondo cuadriculado
plt.tight_layout()


