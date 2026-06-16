# Proyecto cactus
# Utiliza OpenCLIP para vectorizar las imagenes y buscar sus datos
# Muestra la inforamcion de cactus seleccionado
# Autor: Gerardo Figueroa
# Fecha: 15/06/26
import streamlit as st
import os
from PIL import Image
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenCLIPEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# 1. Configuración de la página (Única al inicio)
st.set_page_config(layout="wide")

API_KEY = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = API_KEY

# Inicializar el modelo de lenguaje de Groq
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)

# --- Interfaz de usuario ---
st.header("🌵 Buscador de Cactáceas por Reconocimiento de Imagen (CLIP + RAG)")
st.write("Selecciona una foto. El sistema usará una red neuronal para vectorizarla, identificarla y generar su ficha botánica.")

IMAGE_DIR = "images"

# Verificar la existencia de la carpeta de imágenes
if not os.path.exists(IMAGE_DIR):
    st.error(f"No se encontró la carpeta '{IMAGE_DIR}'. Por favor créala en tu repositorio y sube tus 50 fotos.")
    st.stop()

# Leer las imágenes disponibles
fotos_cactus = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not fotos_cactus:
    st.warning("No se encontraron imágenes en la carpeta 'images/'.")
    st.stop()


# --- FUNCIÓN CRÍTICA: Indexación de Imágenes con CLIP (Cachada para rendimiento) ---
@st.cache_resource
def inicializar_base_datos_visual(lista_fotos):
    """
    Carga el modelo OpenCLIP, extrae los embeddings de todas las fotos
    y las guarda en una base de datos Chroma temporal en memoria.
    """
    # Inicializa el modelo CLIP de OpenAI mediante LangChain
    clip_embeddings = OpenCLIPEmbeddings(model_name="ViT-B-32", checkpoint="openai")
    
    # Preparamos las rutas absolutas de los archivos de imagen
    rutas_imagenes = [os.path.join(IMAGE_DIR, foto) for foto in lista_fotos]
    
    # Creamos metadatos para saber a qué nombre de archivo corresponde cada vector
    metadatos = [{"nombre_archivo": foto} for foto in lista_fotos]
    
    # Creamos la base de datos vectorial Chroma exclusivamente para imágenes
    vectorstore = Chroma.from_images(
        images=rutas_imagenes,
        embedding=clip_embeddings,
        metadatas=metadatos
    )
    return vectorstore, clip_embeddings


# Inicializar la base de datos visual
with st.spinner("Inicializando motor de visión artificial (CLIP)... Esto puede tardar un momento la primera vez."):
    try:
        vectorstore_visual, modelo_clip = inicializar_base_datos_visual(fotos_cactus)
    except Exception as e:
        st.error(f"Error al cargar el modelo de visión: {e}")
        st.stop()


# --- DISEÑO DE LA INTERFAZ ---
col_izq, col_der = st.columns([1, 2])

with col_izq:
    st.subheader("Selección de Imagen")
    foto_seleccionada = st.selectbox("Elige la foto del cactus a reconocer:", sorted(fotos_cactus))
    
    # Mostrar la vista previa de la imagen seleccionada por el usuario
    ruta_seleccionada = os.path.join(IMAGE_DIR, foto_seleccionada)
    st.image(ruta_seleccionada, caption=f"Archivo activo: {foto_seleccionada}", width=250)

with col_der:
    st.subheader("Ficha Botánica Automatizada")
    
    with st.spinner("Analizando patrones visuales y buscando metadatos..."):
        try:
            # 1. El usuario seleccionó una imagen física. Obtenemos su embedding para buscar su "clon" en la BD
            # Buscamos las imágenes más similares (en este caso k=1 traerá la misma imagen indexada)
            resultados_busqueda = vectorstore_visual.similarity_search_by_vector(
                embedding=modelo_clip.embed_image([ruta_seleccionada])[0],
                k=1
            )
            
            if resultados_busqueda:
                # Recuperamos el nombre del archivo guardado en los metadatos del vector más cercano
                archivo_identificado = resultados_busqueda[0].metadata["nombre_archivo"]
                
                # Limpiamos el nombre del archivo para pasárselo como concepto semilla a la IA (ej: "img_cactus_saguaro.png" -> "cactus saguaro")
                nombre_limpio = archivo_identificado.replace(".png", "").replace(".jpg", "").replace(".jpeg", "").replace("img_", "").replace("_", " ")
                
                # 2. Generamos el prompt para Llama utilizando el reconocimiento visual como contexto
                system_prompt = (
                    "Eres una Inteligencia Artificial experta en botánica, biología y cactáceas del mundo.\n"
                    "A partir del nombre del cactus identificado visualmente, genera una ficha informativa profesional, breve y atractiva.\n"
                    "Debes estructurar tu respuesta estrictamente con las siguientes secciones en negritas:\n"
                    "- **Nombre Común y Científico probable**\n"
                    "- **Características físicas y adaptaciones**\n"
                    "- **Región geográfica, hábitat y origen**\n"
                    "- **Un dato curioso o estado de conservación**\n\n"
                    "Confía en tus amplios conocimientos botánicos para redactar el contenido de forma verídica y educativa.\n"
                    "Responde siempre en Español."
                )
                
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "Genera la ficha informativa completa para la cactácea identificada como: {input}"),
                ])
                
                # Ejecución directa con el LLM
                cadena_ia = prompt | llm
                respuesta_final = cadena_ia.invoke({"input": nombre_limpio})
                
                # Mostrar los resultados de la IA en pantalla
                st.success(f"¡Imagen identificada con éxito como patrón correlacionado a '{nombre_limpio}'!")
                st.markdown(respuesta_final.content)
            else:
                st.error("No se pudo correlacionar la firma vectorial de la imagen.")
                
        except Exception as e:
            st.error(f"Hubo un inconveniente en el proceso de análisis multimodal: {e}")