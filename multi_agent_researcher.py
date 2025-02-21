import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
import os

# Inicializar el modelo GPT-4
gpt4_model = None

def create_team_items(issue):
    # Crear agentes
    researcher = Agent(
        role='Investigador',
        goal='Realizar una investigación exhaustiva sobre el tema dado',
        backstory='Eres un investigador experto con un ojo agudo para los detalles',
        verbose=True,
        allow_delegation=False,
        llm=gpt4_model
    )

    redactor = Agent(
        role='Redactor',
        goal='Escribir un artículo detallado y atractivo basado en la investigación, utilizando formato adecuado en markdown',
        backstory='Eres un redactor hábil con experiencia en crear contenido informativo y formatearlo hermosamente en markdown',
        verbose=True,
        allow_delegation=False,
        llm=gpt4_model
    )

    editor = Agent(
        role='Editor',
        goal='Revisar y mejorar el artículo para mayor claridad, precisión, atractivo y formato correcto en markdown',
        backstory='Eres un editor experimentado con un ojo crítico para contenido de calidad y estructura en markdown',
        verbose=True,
        allow_delegation=False,
        llm=gpt4_model
    )

    # Crear tareas
    research_task = Task(
        description=f"Realiza una investigación completa sobre el tema: {issue}. Recopila información clave, estadísticas y opiniones de expertos.",
        agent=researcher,
        expected_output="Un informe de investigación completo sobre el tema dado, incluyendo información clave, estadísticas y opiniones de expertos."
    )

    writing_task = Task(
        description="""Usando la investigación proporcionada, escribe un artículo detallado y atractivo. 
        Asegúrate de tener una estructura adecuada, flujo y claridad. Formatea el artículo usando markdown, incluyendo:
        1. Un título principal (H1)
        2. Subtítulos (H2)
        3. Subapartados donde sea apropiado (H3)
        4. Listas con viñetas o numeradas cuando sea relevante
        5. Énfasis en puntos clave usando texto en negrita o cursiva
        Asegúrate de que el contenido esté bien organizado y sea fácil de leer.""",
        agent=redactor,
        expected_output="Un artículo bien estructurado, detallado y atractivo basado en la investigación proporcionada, formateado en markdown con títulos y subtítulos adecuados."
    )

    editing_task = Task(
        description="""Revisa el artículo para mayor claridad, precisión, atractivo y formato correcto en markdown. 
        Asegúrate de que:
        1. El formato en markdown sea correcto y consistente
        2. Los títulos y subtítulos se usen apropiadamente
        3. El flujo del contenido sea lógico y atractivo
        4. Los puntos clave estén enfatizados correctamente
        Realiza los ajustes y mejoras necesarios en el contenido y el formato.""",
        agent=editor,
        expected_output="Una versión final y pulida del artículo con mayor claridad, precisión, atractivo y formato adecuado en markdown."
    )

    # Crear el equipo
    team = Crew(
        agents=[researcher, redactor, editor],
        tasks=[research_task, writing_task, editing_task],
        verbose=True,
        process=Process.sequential
    )

    return team

# Aplicación Streamlit
st.set_page_config(page_title="Investigador Multi Agente para generar artículos", page_icon="📝")

# CSS personalizado para una mejor apariencia
st.markdown("""
    <style>
    .stApp {
        max-width: 1800px;
        margin: 0 auto;
        font-family: Arial, sans-serif;
    }
    .st-bw {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📝 Generador de Articulos Multi Agente")

# Barra lateral para la clave de la API
with st.sidebar:
    st.header("Configuración")
    api_key = st.text_input("Introduce tu clave de API de OpenAI:", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        gpt4_model = ChatOpenAI(model_name="gpt-4o")
        st.success("¡Clave de API configurada exitosamente!")
    else:
        st.info("Por favor, introduce tu clave de API de OpenAI para continuar.")

# Contenido principal
st.markdown("Genera artículos detallados sobre cualquier tema usando agentes de IA.")

issue = st.text_input("Introduce el tema del artículo:", placeholder="e.g., El impacto de la inteligencia artificial en la salud")

if st.button("Generar artículo"):
    if not api_key:
        st.error("Por favor, introduce tu clave de API de OpenAI en la barra lateral.")
    elif not issue:
        st.warning("Por favor, introduce un tema para el artículo.")
    else:
        with st.spinner("🤖 Los agentes de IA están trabajando en tu artículo..."):
            equipo = create_team_items(issue)
            resultado = equipo.kickoff()
            st.markdown(resultado)

st.markdown("---")
st.markdown("Desarrollado con Langchain y CrewAI ")
