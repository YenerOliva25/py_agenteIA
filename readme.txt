Para la implementación del sistema se utilizaron las siguientes dependencias:

1. Flask: Framework ligero para crear el servidor web y exponer los endpoints de predicción.

pip install flask


2. Flask-CORS: Permite habilitar el intercambio de recursos entre dominios (CORS), necesario para recibir solicitudes desde la aplicación Android.

pip install flask-cors


3. LangChain Community y Core: Librerías utilizadas para la integración modular con modelos de lenguaje y herramientas de IA.

pip install langchain-community
pip install langchain-core


4. Sentence Transformers: Para el manejo de embeddings y procesamiento de texto avanzado.

pip install sentence-transformers


5. Transformers: Librería de Hugging Face para cargar y utilizar modelos de lenguaje o visión preentrenados.

pip install transformers


6. Torch: Backend necesario para ejecutar modelos de aprendizaje profundo.

pip install torch