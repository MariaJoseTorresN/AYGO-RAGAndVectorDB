# RECUPERACIÓN-GENERACIÓN AUMENTADA (RAG) Y BASES DE DATOS VECTORIALES

El proposito de este laboratio es poner a prueba las librerias Lang CHain y Pinecone a través de los siguientes ejercicios:

1. Envio de mensajes a ChatGPT, desde un programa en Python (ExOne).
2. RAG simple usando una base de datos vectorial en memoria (ExTwo).
3. RAG con Pinecone (ExThree).

## Explicación ejercicios

## 1. Envio Prompt y recepción de respuestas de GhatGPT

En este ejercicio se puede evidenciar el uso de la biblioteca "langchain" para construir una cadena de procesamiento de lenguaje natural que aprovecha el modelo de lenguaje de OpenAI (GPT-3) para responder preguntas específicas.
Este ejecicio sirve como punto de partida para proyectos que requieran interacción con modelos de lenguaje de OpenAI para responder preguntas de manera estructurada.

### Configuración Ejercicio 1

Lo principal es configurar la clave de API de OpenAI, mediante el siguiente bloque de código:

```python
os.environ["OPENAI_API_KEY"] = "Tu llave de OpenAI"
```

Reemplaza `Tu llave de OpenAI` con tu propia clave de API de OpenAI.

### Pasos

1. Definición de la Plantilla de Pregunta y Respuesta: La siguiente plantilla define el formato de la entrada y salida esperada; siendo `{question}` un marcador de posición para la pregunta.

```python
template = """Question: {question}

Answer: Let's think step by step."""
```

2. Creación de la Plantilla de Prompt: La siguiente plantilla de prompt permite especificar que la variable de entrada esperada es "question".

```python
prompt = PromptTemplate(template=template, input_variables=["question"])
```

3. Inicialización del Modelo de Lenguaje de OpenAI: Se inicializa el modelo de lenguaje de OpenAI con la clase `OpenAI` de la biblioteca "langchain".

```python
llm = OpenAI()
```
4. Ejecución Cadena de Procesamiento: Con la cadena de procesamiento de lenguaje natural utilizando la plantilla de prompt y el modelo de lenguaje de OpenAI y se da una pregunta específica al modelo. 
La cadena utiliza la plantilla de prompt para estructurar la pregunta y obtener una respuesta.

```python
llm_chain = LLMChain(prompt=prompt, llm=llm)
question = "What is at the core of Popper's theory of science?"
response = llm_chain.run(question)
```
5. Impresión de la Respuesta: La respuesta generada por el modelo de lenguaje de OpenAI se imprime en la consola.

```python
print(response)
```
## 2. RAG simple usando una base de datos vectorial en memoria

El ejecicio sirve como muestra del uso de la biblioteca "langchain" para construir una cadena de procesamiento de lenguaje natural que utiliza el modelo de OpenAI junto con Retrieval-Augmented Generation (RAG) para responder preguntas específicas sobre documentos web.
Este ejercicio es un ejemplo de cómo utilizar Retrieval-Augmented Generation (RAG) junto con un modelo de lenguaje de OpenAI para responder preguntas específicas basadas en documentos web, permitiendo experimentar con diferentes preguntas para explorar la capacidad de recuperación de información relevante del modelo.

### Configuración

Al igual que con el ejercicio anterior, se debe configurar la clave de API de OpenAI.

### Pasos 
1. Configuración de la Carga de Documentos Web: Utilizando `WebBaseLoader` para cargar documentos web desde una URL específica.

```python
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
```

2. División de Texto y Generación de Vectores: El texto se divide en fragmentos para facilitar la generación de vectores y la clase `Chroma` se utiliza para generar vectores a partir de los fragmentos de texto.

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
```

3. Creación de la Cadena de Procesamiento de Lenguaje Natural con RAG: Utilizando el modelo RAG, para que la cadena incluya un retrivador de documentos basado en vectores (Chroma), un formateo del contexto de los documentos, un modelo de lenguaje GPT-3.5-turbo de OpenAI y un parser para manejar la salida de la cadena.

```python
retriever = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```
4. Invocación de la Cadena con una Pregunta Específica e Impresión de la Respuesta: Al proporcionar una pregunta específica a la cadena de procesamiento, para que, con el modelo RAG, recupere la información relevante del contexto de los documentos web y genere una respuesta que se mostrara en la consola.

```python
response = rag_chain.invoke("What is Task Decomposition?")
print(response)
```

## 3. RAG con Pinecone

El ejercicio muestra cómo utilizar la biblioteca "langchain" junto con Pinecone para realizar búsquedas de documentos basadas en similitud de embeddings.
Este ejercicio es un buen ejemplo de cómo utilizar Pinecone junto con embeddings generados por el modelo de lenguaje de OpenAI para realizar búsquedas de documentos basadas en similitud de texto. 

### Configuración

Al igual que con los dos ejrccos anteriores se debe configurar las claves de API de OpenAI, pero en este caso tambien la API de Pinecone.

```python
os.environ["OPENAI_API_KEY"] = "Tu llave de OpenAI"
os.environ["PINECONE_API_KEY"] = "Tu llave de Pinecone"
os.environ["PINECONE_ENV"] = "gcp-starter"
```
Reemplaza las claves de API con las tuyas.

### Pasos

1. Carga de Texto y Generación de Embeddings: Utilizando `TextLoader` para cargar un documento de texto ("awedfirstpaper.txt" en este caso), para luego generar embeddings con el modelo de lenguaje de OpenAI.

```python
loader = TextLoader("awedfirstpaper.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
```
2. Configuración e Inicialización de Pinecone: Con Pinecone se realizan búsquedas de similitud de embeddings, inicializando el Pinecone con las claves de API y verificando si el índice ya existe; si no existe, se crea un nuevo índice con dimensiones específicas (1536 en este caso, que coincide con las dimensiones del modelo de lenguaje de OpenAI utilizado).

```python
import pinecone

# initialize pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV"),
)

index_name = "langchain-demo"

# Verificar si el índice ya existe, si no, crearlo
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, metric="cosine", dimension=1536)
```

3. Indexación de Documentos en Pinecone: Para habilitar la búsqueda de similitud de embeddings.

```python
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
```

4. Búsqueda de Documentos Similares: Utilizando una consulta específica, siendo la respuesta el documento más similar a la consulta, para mostrarse en la consola.

```python
query = "What is a distributed pointcut"
docs = docsearch.similarity_search(query)

print(docs[0].page_content)
```

## Resumen
Los ejercicios desarrollados demuestran la versatilidad de langchain para integrar modelos de lenguaje de OpenAI, responder preguntas y realizar generación de texto, incluyendo el uso de la arquitectura Retrieval-Augmented Generation (RAG) para contextualizar respuestas en documentos web; además, se aprovecha la eficacia de Pinecone para indexar y buscar documentos basados en similitud de embeddings generados por modelos de lenguaje. 
Se subraya la importancia de configurar adecuadamente las claves de API y ambientales, así como la manipulación estructurada de texto para abordar tareas específicas en el procesamiento de lenguaje natural y la recuperación de información.