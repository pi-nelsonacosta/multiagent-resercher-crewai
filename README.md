# 📝 Generador de Artículos Multi-Agente con IA

¡Genera artículos detallados y estructurados automáticamente usando agentes de IA colaborativos!

## 🚀 Descripción

Esta aplicación utiliza **CrewAI**, **Langchain** y **GPT-4** para crear artículos bien estructurados mediante un equipo de tres agentes especializados:
1. **Investigador**: Recopila información clave y datos relevantes
2. **Redactor**: Elabora el contenido con formato markdown profesional
3. **Editor**: Refina el artículo para máxima calidad y claridad

## ✨ Características Principales

- 🔍 Investigación automatizada de temas
- ✍️ Redacción con formato markdown estructurado
- ✅ Edición inteligente para calidad y coherencia
- 🎨 Interfaz intuitiva con Streamlit
- 🤖 Flujo de trabajo secuencial entre agentes

## 📦 Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/pi-nelsonacosta/multiagent-resercher-crewai.git
```

2. Crea un entorno virtual y activalo:
```bash
python -m venv venv
source venv/bin/activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

🖥️ Uso
Inicia la aplicación:
```bash
streamlit run multi_agent_researcher.py
```

🛠️ Tecnologías Utilizadas

**Streamlit:** Interfaz web interactiva

**CrewAI:** Coordinación de agentes de IA

**Langchain:** Integración con modelos de lenguaje

**Markdown:** Formateo estructurado de contenido

📋 Requisitos
- Python 3.9+
- Cuenta de OpenAI con créditos

📌 Dependencias
- streamlit
- crewai
- langchain-openai
- python-dotenv


🔄 Flujo de Trabajo
- Investigación → 2. Redacción → 3. Edición


📝 Licencia
Este proyecto está bajo la licencia MIT.