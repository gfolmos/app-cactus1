# Proyecto cactus
# Utiliza OpenCLIP para vectorizar las imagenes y buscar sus datos
# Muestra la inforamcion de cactus seleccionado
# Autor: Gerardo Figueroa
# Fecha: 15/06/26
import streamlit as st
import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModel
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
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
    st.error(f"No se encontró la carpeta '{IMAGE_DIR}'. Por favor créala en tu repositorio y sube tus fotos.")
    st.stop()

# Leer las imágenes disponibles
fotos_cactus = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not fotos_cactus:
    st.warning("No se encontraron imágenes en la carpeta 'images/'.")
    st.stop()


# --- FUNCIÓN CRÍTICA: Inicializar el Modelo CLIP nativo y indexar imágenes ---
@st.cache_resource
def inicializar_modelo_y_vectores(lista_fotos):
    """
    Carga el procesador y modelo CLIP nativo de Hugging Face, extrae los embeddings 
    de todas las fotos y los indexa en Chroma usando vectores puros.
    """
    # Cargamos el modelo de visión oficial de OpenAI en Hugging Face
    procesador = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    modelo_vision = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    
    # Inicializamos Chroma de forma manual (sin pasarle un objeto embedding de LangChain)
    # Usaremos una colección temporal en memoria
    import chromadb
    cliente_chroma = chromadb.EphemeralClient()
    coleccion = cliente_chroma.create_collection(name="cactus_collection")
    
    # Indexar cada foto
    for foto in lista_fotos:
        ruta_completa = os.path.join(IMAGE_DIR, foto)
        imagen = Image.open(ruta_completa).convert("RGB")
        
        # Procesar imagen y extraer características (Vector)
        inputs = procesador(images=imagen, return_tensors="pt")
        with torch.no_grad():
            outputs = modelo_vision(**inputs)
        
        # Convertir el tensor a una lista de flotantes (Embedding)
        embedding = outputs.pooler_output.flatten().tolist()
        
        # Guardar en la base de datos de Chroma
        coleccion.add(
            embeddings=[embedding],
            ids=[foto],
            metadatas=[{"nombre_archivo": foto}]
        )
        
    return procesador, modelo_vision, coleccion


# Inicializar el motor de visión
with st.spinner("Inicializando motor de visión artificial nativo (CLIP)... Esto puede tardar un momento la primera vez."):
    try:
        procesador_clip, modelo_clip, coleccion_visual = inicializar_modelo_y_vectores(fotos_cactus)
    except Exception as e:
        st.error(f"Error al cargar el modelo de visión: {e}")
        st.stop()


# --- DISEÑO DE LA INTERFAZ ---
col_izq, col_der = st.columns([1, 2])

with col_izq:
    st.subheader("Selección de Imagen")
    foto_seleccionada = st.selectbox("Elige la foto del cactus a reconocer:", sorted(fotos_cactus))
    
    # Mostrar la vista previa de la imagen seleccionada
    ruta_seleccionada = os.path.join(IMAGE_DIR, foto_seleccionada)
    st.image(ruta_seleccionada, caption=f"Archivo activo: {foto_seleccionada}", width=250)

with col_der:
    st.subheader("Ficha Botánica Automatizada")
    
    with st.spinner("Analizando patrones visuales con CLIP..."):
        try:
            # 1. Extraer el embedding de la foto que el usuario acaba de seleccionar
            imagen_actual = Image.open(ruta_seleccionada).convert("RGB")
            inputs_actual = procesador_clip(images=imagen_actual, return_tensors="pt")
            
            with torch.no_grad():
                outputs_actual = modelo_clip(**inputs_actual)
            embedding_actual = outputs_actual.pooler_output.flatten().tolist()
            
            # 2. Buscar en Chroma el vector más cercano (k=1)
            busqueda = coleccion_visual.query(
                query_embeddings=[embedding_actual],
                n_results=1
            )
            
            if busqueda and busqueda['metadatas'][0]:
                # Recuperar el nombre del archivo del metadato del clon más cercano
                archivo_identificado = busqueda['metadatas'][0][0]["nombre_archivo"]
                
                # Limpiar el nombre del archivo para dárselo a Llama como concepto
                nombre_limpio = archivo_identificado.replace(".png", "").replace(".jpg", "").replace(".jpeg", "").replace("img_", "").replace("_", " ")
                
                # 3. Generar la ficha técnica con Llama 3.3
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
                
                cadena_ia = prompt | llm
                respuesta_final = cadena_ia.invoke({"input": nombre_limpio})
                
                # Mostrar resultados
                st.success(f"¡Imagen identificada con éxito! Coincidencia vectorial: '{nombre_limpio}'")
                st.markdown(respuesta_final.content)
            else:
                st.error("No se pudo correlacionar la firma vectorial de la imagen.")
                
        except Exception as e:
            st.error(f"Hubo un inconveniente en el proceso de análisis multimodal: {e}")