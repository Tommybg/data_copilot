import streamlit as st

# Configure page settings
st.set_page_config(page_title="Data Analyzer", layout="wide")

# Data manipulation and visualization imports
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import os
import json
from dotenv import load_dotenv
import tempfile
import io

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentType
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage

# Excel imports
import openpyxl
import xlrd

# Audio imports - with error handling
try:
    from audio_recorder_streamlit import audio_recorder
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    st.warning("Audio recorder not available. Install with: pip install audio-recorder-streamlit")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("OpenAI not available. Install with: pip install openai")

# Load environment variables
load_dotenv()

# Configure OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.info("You can add your API key to a .env file in the same directory as this script.")
    st.stop()

# System prompt template
SYSTEM_PROMPT = """You are a data analysis assistant that helps government users understand their data through visualizations and analysis, and identify valuable insights. 
When working with the provided Data, follow these guidelines:

1. ANALYSIS APPROACH:
   - First, understand what the user is asking for.
   - Check if the data is suitable for the requested analysis.
   - Determine if a visualization would be helpful, if not suggest something better and make it. 
   - Provide clear, concise explanations.

2. VISUALIZATION GUIDELINES:
   When creating visualizations, you must return TWO parts:
   a) A text explanation/analysis of the insights
   b) A JSON visualization object formatted exactly like this example:

For bar charts:
{{"type": "bar", "title": "Title", "data": {{"labels": ["A", "B"], "values": [{{"data": [1, 2], "label": "Series"}}]}}, "xAxis": {{"label": "X"}}, "yAxis": {{"label": "Y"}}}}

For line charts:
{{"type": "line", "title": "Title", "data": {{"xValues": ["2021", "2022"], "yValues": [{{"data": [1, 2], "label": "Series"}}]}}, "xAxis": {{"label": "X"}}, "yAxis": {{"label": "Y"}}}}

For pie charts:
{{"type": "pie", "title": "Title", "data": [{{"label": "A", "value": 30}}, {{"label": "B", "value": 70}}]}}

For scatter plots:
{{"type": "scatter", "title": "Title", "data": {{"series": [{{"data": [{{"x": 1, "y": 2, "id": 1}}], "label": "Series"}}]}}, "xAxis": {{"label": "X"}}, "yAxis": {{"label": "Y"}}}}

For horizontal bar charts:
{{"type": "horizontal_bar", "title": "Title", "data": {{"labels": ["A", "B"], "values": [{{"data": [1, 2], "label": "Series"}}]}}, "xAxis": {{"label": "X"}}, "yAxis": {{"label": "Y"}}}}

For histograms:
{{"type": "histogram", "title": "Title", "data": {{"values": [{{"data": [1, 2, 3, 4, 5], "label": "Series"}}]}}}, "xAxis": {{"label": "Value"}}, "yAxis": {{"label": "Frequency"}}}}

For boxplots:
{{"type": "boxplot", "title": "Title", "data": {{"values": [{{"data": [1, 2, 3, 4, 5], "label": "Series"}}]}}}, "xAxis": {{"label": "Category"}}, "yAxis": {{"label": "Value"}}}}

3. CHOOSING VISUALIZATIONS:
    - Bar Charts: Use for comparing different categories
    - Line Charts: Use for showing trends over time  
    - Pie Charts: Use for showing proportions of a whole
    - Scatter Plots: Use for analyzing relationships between two variables
    - Horizontal Bar Charts: Use when category names are long
    - Histogram: Use for distribution of continuous variables
    - Boxplot: Use for summarizing data distributions and identifying outliers

4. REASONING: 
    - Choose the appropriate visualization based on the data and user request
    - Don't hallucinate any data or results
    - Use best practices for design in visualizations
    - Maintain consistent design and beautiful visualizations

5. RESPONSE FORMAT:
   Always structure your response as:
   1. Brief explanation of the analysis
   2. Key insights from the data
   3. The visualization JSON object if applicable

Remember:
- Always respond in Spanish (Colombia)
- Keep explanations concise but informative
- Always validate data before creating visualizations
- Handle missing or invalid data appropriately
- Include proper labels and titles in visualizations
- Format numbers for readability
- Do not use inappropriate language, stick to data analysis
- If asked who you are, say "Soy Data Copilot tu copilot para el análisis de datos, diseñado para obtener insights, análisis y propuestas de valor por medio de los datos dados"
- Remember you have memory and can see the past 5 messages for context
"""

# Create uploads directory if it doesn't exist
UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(exist_ok=True)

class ConversationManager:
    def __init__(self, window_size=5):
        """Initialize conversation manager with specified memory window"""
        # Initialize memory for chat history
        self.memory = ConversationBufferWindowMemory(
            k=window_size,
            return_messages=True,
            memory_key="chat_history",
            input_key="input"
        )
        
    def add_message(self, role: str, content: str):
        """Add a message to memory and ensure it's synced with session state"""
        try:
            if role == "human":
                self.memory.chat_memory.add_message(HumanMessage(content=content))
            elif role == "ai":
                self.memory.chat_memory.add_message(AIMessage(content=content))
                
            # Sync memory with session state messages
            if 'messages' in st.session_state:
                # Update session state messages to match memory
                messages = self.get_chat_history()
                st.session_state.messages = [
                    {
                        "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                        "content": msg.content
                    }
                    for msg in messages[-10:]  # Keep last 10 messages to prevent overflow
                ]
        except Exception as e:
            st.error(f"Error adding message to memory: {str(e)}")
    
    def get_chat_history(self):
        """Get the current chat history"""
        return self.memory.chat_memory.messages
    
    def clear_memory(self):
        """Clear both the conversation memory and session state messages"""
        self.memory.clear()
        if 'messages' in st.session_state:
            st.session_state.messages = []

def create_plotly_visualization(vis_data):
    """Create Plotly visualizations based on generated graph data"""
    if not isinstance(vis_data, dict):
        raise ValueError("Invalid visualization data format")
    
    if 'type' not in vis_data or 'data' not in vis_data:
        raise ValueError("Missing required visualization parameters")
        
    try:
        graph_type = vis_data['type']
        data = vis_data['data']
        
        # Set default color scheme
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
        
        fig = go.Figure()
        
        if graph_type == 'bar':
            if not isinstance(data.get('values', []), list) or not isinstance(data.get('labels', []), list):
                raise ValueError("Invalid data format for bar chart")
            for idx, series in enumerate(data.get('values', [])):
                fig.add_trace(go.Bar(
                    x=data['labels'],
                    y=series.get('data', []),
                    name=series.get('label', f'Series {idx+1}'),
                    marker_color=colors[idx % len(colors)]
                ))
            fig.update_layout(barmode='group')
            
        elif graph_type == 'horizontal_bar':
            if not isinstance(data.get('values', []), list) or not isinstance(data.get('labels', []), list):
                raise ValueError("Invalid data format for horizontal bar chart")
            for idx, series in enumerate(data.get('values', [])):
                fig.add_trace(go.Bar(
                    y=data['labels'],
                    x=series.get('data', []),
                    orientation='h',
                    name=series.get('label', f'Series {idx+1}'),
                    marker_color=colors[idx % len(colors)]
                ))
            fig.update_layout(barmode='group')
            
        elif graph_type == 'line':
            if not isinstance(data.get('xValues', []), list) or not isinstance(data.get('yValues', []), list):
                raise ValueError("Invalid data format for line chart")
            for idx, series in enumerate(data.get('yValues', [])):
                fig.add_trace(go.Scatter(
                    x=data['xValues'],
                    y=series.get('data', []),
                    mode='lines+markers',
                    name=series.get('label', f'Series {idx+1}'),
                    line=dict(color=colors[idx % len(colors)])
                ))
                
        elif graph_type == 'pie':
            if not isinstance(data, list):
                raise ValueError("Invalid data format for pie chart")
            fig = go.Figure(data=[go.Pie(
                labels=[item.get('label', '') for item in data],
                values=[item.get('value', 0) for item in data],
                marker=dict(colors=colors[:len(data)])
            )])
            
        elif graph_type == 'scatter':
            if not isinstance(data.get('series', []), list):
                raise ValueError("Invalid data format for scatter plot")
            for idx, series in enumerate(data.get('series', [])):
                x_vals = [point.get('x', 0) for point in series.get('data', [])]
                y_vals = [point.get('y', 0) for point in series.get('data', [])]
                fig.add_trace(go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='markers',
                    name=series.get('label', f'Series {idx+1}'),
                    marker=dict(color=colors[idx % len(colors)])
                ))
        
        elif graph_type == 'histogram':
            values = data.get('values', [])
            if isinstance(values, list) and len(values) > 0:
                # Handle both old and new format
                if isinstance(values[0], dict):
                    # New format with series
                    for idx, series in enumerate(values):
                        fig.add_trace(go.Histogram(
                            x=series.get('data', []),
                            name=series.get('label', f'Series {idx+1}'),
                            marker_color=colors[idx % len(colors)],
                            opacity=0.75 if len(values) > 1 else 1
                        ))
                else:
                    # Old format - direct values
                    fig.add_trace(go.Histogram(
                        x=values,
                        name=data.get('label', 'Histogram'),
                        marker_color=colors[0],
                        opacity=1
                    ))
                if len(values) > 1:
                    fig.update_layout(barmode='overlay')
            
        elif graph_type in ['box', 'boxplot']:
            values = data.get('values', [])
            if isinstance(values, list) and len(values) > 0:
                for idx, series in enumerate(values):
                    fig.add_trace(go.Box(
                        y=series.get('data', []),
                        name=series.get('label', f'Series {idx+1}'),
                        marker_color=colors[idx % len(colors)],
                        boxpoints='outliers',
                        boxmean=True
                    ))
        else:
            raise ValueError(f"Unsupported graph type: {graph_type}")

        # Enhanced layout configuration
        fig.update_layout(
            template="plotly_white",
            title=dict(
                text=vis_data.get('title', ''),
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            xaxis_title=dict(
                text=vis_data.get('xAxis', {}).get('label', ''),
                font=dict(size=14)
            ),
            yaxis_title=dict(
                text=vis_data.get('yAxis', {}).get('label', ''),
                font=dict(size=14)
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12)
            ),
            margin=dict(l=50, r=50, t=60, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add grid lines
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        return fig
        
    except Exception as e:
        raise ValueError(f"Error creating visualization: {str(e)}")

def clean_dataframe(df):
    """Clean and prepare dataframe for analysis"""
    try:
        # Remove completely empty rows and columns
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        # Try to convert numeric columns to numeric type
        for col in df.columns:
            try:
                # Replace common non-numeric characters
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace('$', '', regex=False)
                    df[col] = df[col].str.replace(',', '', regex=False)
                    df[col] = df[col].str.replace('%', '', regex=False)
                # Try to convert to numeric
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except:
                continue
        
        return df
    except Exception as e:
        print(f"Error cleaning dataframe: {e}")
        return df

@st.cache_data
def load_file(uploaded_file):
    """Load and process uploaded file with caching"""
    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        if file_extension == '.csv':
            try:
                # Try different encodings
                encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error with encoding {encoding}: {e}")
                        continue
                
                if df is None:
                    st.error("No se pudo leer el archivo CSV con ninguna codificación conocida")
                    return None
                    
                if df.empty:
                    st.error("El archivo CSV está vacío")
                    return None
                    
                return clean_dataframe(df)
                
            except Exception as e:
                st.error(f"Error al leer el archivo CSV: {str(e)}")
                return None
                
        elif file_extension in ['.xlsx', '.xls']:
            try:
                # Try to read all sheets
                all_sheets = pd.read_excel(uploaded_file, sheet_name=None)
                
                if len(all_sheets) > 1:
                    # If multiple sheets, let user select one
                    sheet_name = st.selectbox(
                        "Selecciona una hoja del archivo Excel:",
                        list(all_sheets.keys())
                    )
                    df = all_sheets[sheet_name]
                else:
                    # If only one sheet, use it
                    df = pd.read_excel(uploaded_file)
                
                if df.empty:
                    st.error("El archivo Excel está vacío")
                    return None
                    
                return clean_dataframe(df)
                
            except Exception as e:
                st.error(f"Error al leer el archivo Excel: {str(e)}")
                return None
        else:
            st.error(f"Formato de archivo no soportado. Por favor, usa archivos CSV o Excel (.xlsx, .xls)")
            return None
            
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return None

def initialize_agent(df, conversation_manager):
    """Initialize LangChain agent with memory"""
    try:
        # Basic configuration for ChatOpenAI
        llm = ChatOpenAI(
            model="gpt-4",  # Using standard model name
            temperature=0,
            streaming=True,
            api_key=OPENAI_API_KEY
        )
        
        # Create the agent with the configured LLM
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            handle_parsing_errors=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix=SYSTEM_PROMPT,
            allow_dangerous_code=True  # Required for newer versions
        )
        
        return agent
    except Exception as e:
        st.error(f"Error al inicializar el modelo de IA: {str(e)}")
        print(f"Detailed error: {e}")  # For debugging
        return None

def process_agent_response(response):
    """Process the agent's response to extract visualization data if present"""
    try:
        # Remove any ```json and ``` tags from the response
        clean_response = str(response).replace('```json', '').replace('```', '')
        
        # Check if response contains JSON visualization data
        start_idx = clean_response.find('{')
        end_idx = clean_response.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = clean_response[start_idx:end_idx]
            # Clean up any potential markdown formatting
            json_str = json_str.strip('`').strip()
            
            try:
                vis_data = json.loads(json_str)
                if isinstance(vis_data, dict) and 'type' in vis_data and 'data' in vis_data:
                    text_part = clean_response[:start_idx].strip()
                    return text_part if text_part else clean_response, vis_data
            except json.JSONDecodeError:
                pass
        
        return clean_response, None
    except Exception as e:
        print(f"Error processing response: {str(e)}")
        return str(response), None

def has_valid_data():
    """Check if there's valid data in the session state"""
    return (
        'df' in st.session_state 
        and st.session_state.df is not None 
        and isinstance(st.session_state.df, pd.DataFrame) 
        and not st.session_state.df.empty
    )

# Audio Feature Management 
def create_hidden_audio_player(audio_bytes):
    """Create a hidden audio player for autoplay"""
    try:
        import base64
        # Ensure we're dealing with bytes
        if isinstance(audio_bytes, str):
            audio_bytes = audio_bytes.encode()
            
        # Properly encode to base64
        b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return f"""
            <audio autoplay style="display: none;">
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
        """
    except Exception as e:
        print(f"Error creating hidden audio player: {str(e)}")
        return ""

def process_audio_to_text(audio_bytes):
    """Convert audio to text using OpenAI's Whisper model"""
    if not OPENAI_AVAILABLE:
        st.warning("OpenAI not available for audio processing")
        return None
        
    try:
        # Initialize client with explicit API key
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Save audio bytes to a temporary file
        temp_audio_path = "temp_audio.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)
        
        try:
            with open(temp_audio_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    response_format="text"
                )
            return transcription
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    except Exception as e:
        print(f"Error processing audio to text: {str(e)}")
        st.error(f"Error processing audio: {str(e)}")
        return None

def text_to_speech(text):
    """Convert text to speech using OpenAI's TTS API"""
    if not OPENAI_AVAILABLE:
        return None
        
    try:
        # Initialize client with explicit API key
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Limit text length for TTS
        if len(text) > 4000:
            text = text[:4000] + "..."
        
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        
        return response.content
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        return None

# CSS Styles
st.markdown("""
    <style>
    /* Hide bootstrap.min.css.map error */
    @import url('bootstrap.min.css') screen and (min-width: 0px);
    
    /* Center all sidebar content */
    .stSidebar .stSidebar-content {
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
        text-align: center;
    }

    /* Center elements in sidebar */
    .css-1kyxreq.e115fcil2 {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
    }

    /* Adjust button spacing and centering */
    .stButton > button {
        display: block;
        margin: 5px auto !important;
        width: 80%;
    }

    div[data-testid="stAudioRecorder"] {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1rem 0;
        width: 100%;
    }
            
    /* Center the drag and drop text */
    [data-testid="stFileUploadDropzone"] {
        text-align: center;
    }
    
    /* Hide deployment button */
    .stDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'conversation_manager' not in st.session_state:
    st.session_state.conversation_manager = ConversationManager(window_size=5)

# Sidebar content
with st.sidebar:
    st.markdown("""
    **Puedes preguntarme cosas como:**
    - Muéstrame un gráfico de barras del gasto público por ministerio
    - ¿Cuál es la tendencia de ingresos fiscales a lo largo del tiempo?
    - Crea un gráfico pie de la distribución del presupuesto
    """)
    st.markdown("***")
                
    uploaded_file = st.file_uploader("Sube tu documento acá", 
                                    type=['csv', 'xlsx', 'xls'],
                                    help="Soporta archivos CSV y Excel (.xlsx, .xls)")
    
    if st.button("Borrar Historial Chat"):
        if 'conversation_manager' in st.session_state:
            st.session_state.conversation_manager.clear_memory()
        st.session_state.messages = []
        st.session_state.df = None
        if 'agent' in st.session_state:
            del st.session_state.agent
        st.rerun()
    
    if st.button("Subir nuevo archivo"):
        if 'conversation_manager' in st.session_state:
            st.session_state.conversation_manager.clear_memory()
        st.session_state.messages = []
        st.session_state.df = None
        if 'agent' in st.session_state:
            del st.session_state.agent
        st.session_state['uploaded_file'] = None
        st.rerun() 
    
    # Add the audio recorder if available
    audio_bytes = None
    if AUDIO_AVAILABLE:
        audio_bytes = audio_recorder(
            text="", 
            icon_size="2x", 
            recording_color="red", 
            neutral_color="#3399ff", 
            key="audio_recorder"
        )

# Main content area
st.title("📊 Data Copilot")

# Initialize user_input
user_input = None

# Handle text input
text_input = st.chat_input("Analicemos tus datos ...")
if text_input and text_input.strip():
    user_input = text_input

# Handle audio input
if audio_bytes is not None and AUDIO_AVAILABLE:
    with st.spinner('Procesando audio...'):
        transcribed_text = process_audio_to_text(audio_bytes)
        if transcribed_text and transcribed_text.strip():
            user_input = transcribed_text
            st.info(f"Transcripción: {transcribed_text}")

# Handle file upload and data display
if uploaded_file:
    try:
        if not has_valid_data() or st.session_state.get('last_uploaded_file') != uploaded_file.name:
            with st.spinner('Cargando archivo...'):
                df = load_file(uploaded_file)
                if df is not None:
                    st.session_state.df = df
                    st.session_state.last_uploaded_file = uploaded_file.name
                    st.success("Archivo subido exitosamente!")
                else:
                    st.error("Error al cargar el archivo")
                    st.stop()
        
        # Data preview and chat interface
        with st.expander("Previsualización Datos", expanded=True):
            st.dataframe(st.session_state.df.head(6))
            st.write(f"**Filas:** {len(st.session_state.df)}, **Columnas:** {len(st.session_state.df.columns)}")
        
        st.markdown("---")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("visualization"):
                    st.plotly_chart(message["visualization"], use_container_width=True)
        
        # Process user input
        if user_input:
            # Add human message to memory
            st.session_state.conversation_manager.add_message("human", user_input)
            
            # Display user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message(name="user", avatar="👨‍💼"):
                st.markdown(user_input)
            
            # Get and display assistant response
            with st.chat_message(name="assistant", avatar="🧠"):
                if 'agent' not in st.session_state or st.session_state.agent is None:
                    with st.spinner('Inicializando el asistente...'):
                        st.session_state.agent = initialize_agent(
                            st.session_state.df,
                            st.session_state.conversation_manager
                        )
                        if st.session_state.agent is None:
                            st.error("No se pudo inicializar el asistente. Por favor, verifica tu API key y conexión.")
                            st.stop()
                
                try:
                    with st.spinner('Procesando tu consulta...'):
                        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                        
                        # Prepare input with context
                        agent_input = {
                            "input": user_input,
                            "chat_history": st.session_state.conversation_manager.get_chat_history()
                        }
                        
                        response = st.session_state.agent.run(agent_input, callbacks=[st_callback])
                    
                    # Add assistant response to memory
                    st.session_state.conversation_manager.add_message("ai", str(response))
                    
                    # Process response for visualization
                    text_response, vis_data = process_agent_response(response)
                    
                    # Display text response
                    if text_response:
                        st.markdown(text_response)
                    
                    # Generate and play audio response if available
                    if OPENAI_AVAILABLE:
                        try:
                            with st.spinner('Generando respuesta de audio...'):
                                audio_response = text_to_speech(text_response or str(response))
                                if audio_response:
                                    hidden_player = create_hidden_audio_player(audio_response)
                                    if hidden_player:
                                        st.markdown(hidden_player, unsafe_allow_html=True)
                        except Exception as e:
                            st.warning("No se pudo generar la respuesta de audio", icon="⚠️")
                            print(f"Error playing audio response: {str(e)}")
                                    
                    # Create and display visualization if present
                    visualization_fig = None
                    if vis_data:
                        try:
                            with st.spinner('Generando visualización...'):
                                visualization_fig = create_plotly_visualization(vis_data)
                                if visualization_fig:
                                    st.plotly_chart(visualization_fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error al crear la visualización: {str(e)}")
                            print(f"Visualization error details: {e}")
                    
                    # Add message to session state
                    message_data = {
                        "role": "assistant",
                        "content": text_response or str(response)
                    }
                    if visualization_fig:
                        message_data["visualization"] = visualization_fig
                    
                    st.session_state.messages.append(message_data)
                    
                except Exception as e:
                    error_message = f"Error al procesar tu pregunta: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })
                    print(f"Agent error details: {e}")
                    
    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
        st.session_state.df = None
        if 'agent' in st.session_state:
            del st.session_state.agent
        print(f"File processing error: {e}")
        
elif not has_valid_data():
    st.info("👈 Por favor sube un archivo CSV para iniciar")
    # Welcome message
    st.markdown("""
    ### ¡Bienvenido a Data Copilot! 🚀
    
    Soy tu asistente inteligente para análisis de datos. Puedo ayudarte a:
    
    - 📊 **Crear visualizaciones** (gráficos de barras, líneas, pie, scatter, etc.)
    - 🔍 **Analizar tendencias** y patrones en tus datos
    - 📈 **Generar insights** valiosos para la toma de decisiones
    - 🎯 **Responder preguntas** específicas sobre tu información
    
    **Para comenzar:** Sube un archivo CSV usando el botón en la barra lateral.
    """)
    
    # Clear any residual data
    if 'df' in st.session_state:
        del st.session_state.df
    if 'agent' in st.session_state:
        del st.session_state.agent