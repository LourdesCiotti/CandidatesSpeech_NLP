"""
Bulrrich
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
import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'plotly_mimetype'

# Cargar el modelo en español
nlp = spacy.load('es_core_news_sm')
# Descargar los recursos necesarios de nltk
nltk.download('stopwords')
nltk.download('punkt')

# Ruta al archivo de texto
archivo = r'C:\Users\Lufly\Desktop\Proyectos\bullrich.txt'
# Leer el contenido del archivo
with open(archivo, 'r', encoding='utf-8') as file:
    texto = file.read()

# Eliminar números y signos de puntuación del texto
#Sumar para que elimine signo de admiracion
translator = str.maketrans('', '', string.digits + string.punctuation + '!' + '¡')
texto = texto.translate(translator)
# Tokenización: Divide el texto en palabras
words = word_tokenize(texto)
# Convertir todo el texto a minúscula
texto = texto.lower()

# Procesar el texto con spaCy
doc = nlp(texto)


# Identificar y mostrar artículos (DET) y preposiciones (ADP) antes de eliminarlos
palabras_a_eliminar = [token.text for token in doc if (token.pos_ in ['ADP', 'DET'] or token.text.lower() in ['no', 'les', 'ser', 'vez', 'donde','son', 'está', 'hay','es','se','también', 'nos', 'tiene','como', 'cuando', 'eso', 'y', 'de', 'porque', 'que', 'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'lo', 'le', 'uno', 'me', 'va', 'ahí', 'esto', 'sido', 'esto', 'córdoba']) and token.text.lower() != 'sin' and token.text.lower() != 'contra']
# Eliminar las palabras definidas en palabras_a_eliminar del texto
texto_filtrado = ' '.join([word.lower() for word in words if word.lower() not in palabras_a_eliminar])
# Lematización
# Utilizar una expresión regular para eliminar letras sueltas
texto_preprocesado = re.sub(r'\s+[A-Za-z]\s+', ' ', texto_filtrado)
# doc_lematizado = nlp(texto_limpio)  # Procesar el texto filtrado con lematización
# lematizacion = [token.lemma_ for token in doc_lematizado]



# Realizar el análisis de sentimientos en el texto preprocesado
sentimientos = TextBlob(texto_preprocesado)
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
print("Texto Preprocesado:", texto_preprocesado)
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
frecuencia_palabras = Counter(texto_preprocesado.split())

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



words_2 = word_tokenize(texto_preprocesado)
# Generar bigramas
bigrams = list(nltk.bigrams(words_2))

# Crear un gráfico de palabras
graph = nx.Graph()
for bigram in bigrams:
    graph.add_edge(bigram[0], bigram[1])

# Calcular la centralidad de grado de cada palabra
centralidades = nx.degree_centrality(graph)

# Imprimir las palabras con mayor centralidad
for palabra, centralidad in centralidades.items():
    print(palabra, centralidad)

# Calcular el tamaño de los nodos basado en la suma de centralidades
node_size = 100
# Crear un objeto de figura de Plotly
fig = go.Figure()

# Crear un layout para el gráfico
layout = go.Layout(
    showlegend=False,
    hovermode='closest'
)

# Agregar nodos al gráfico
for nodo, centralidad in centralidades.items():
    fig.add_trace(go.Scatter(
        x=[],
        y=[],
        text=nodo,
        mode='markers',
        marker=dict(size=centralidad * node_size, line_width=2)
    ))

# Agregar bordes (conexiones) entre nodos
for edge in graph.edges():
    x0, y0 = edge[0], edge[1]
    fig.add_trace(go.Scatter(x=[x0, y0], y=[x0, y0], mode='lines'))

fig.update_layout(layout)

# Guardar el gráfico como una imagen
fig.write_image("grafo.png")

# Aviso de que el gráfico se ha guardado
print("Gráfico guardado como grafo.png")
