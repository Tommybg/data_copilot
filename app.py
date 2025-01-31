import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import os
import json
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentType
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from audio_recorder_streamlit import audio_recorder
from openai import OpenAI
from pathlib import Path
import os



# Load environment variables
load_dotenv()

# Configure OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    st.info("You can add your API key to a .env file in the same directory as this script.")
    st.stop()

# Configure page settings
st.set_page_config(page_title="Data Analyzer", layout="wide")

# System prompt template (Your existing SYSTEM_PROMPT here)
# System prompt template
SYSTEM_PROMPT = """ You are a data analysis assistant that helps governement users understand their data through visualizations and analysis, and identify valuable insights. 
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
{{"type": "histogram", "title": "Title", "data": {{"values": [1, 2, 3, 4, 5], "label": "Series"}}}, "xAxis": {{"label": "Value"}}, "yAxis": {{"label": "Frequency"}}}}

For boxplots:
{{"type": "boxplot", "title": "Title", "data": {{"values": [{{"data": [1, 2, 3, 4, 5], "label": "Series"}}]}}}, "xAxis": {{"label": "Category"}}, "yAxis": {{"label": "Value"}}}}

3. CHOOSING VISUALIZATIONS:
    - Bar Charts
        Use Case: Bar charts are ideal for comparing different categories or groups. They clearly show differences in magnitude between discrete categories, making it easy to visualize relative sizes, such as sales figures for different products or survey responses across demographics. 
        **When to use:** Use when you want to compare quantities across different categories.
        **Keywords (Spanish):** "comparar", "gráfico de barras", "categorías", "diferencias"

    - Line Charts
        Use Case: Line charts are best suited for displaying trends over time. They track changes across continuous data points, making them useful for showing how variables evolve, such as stock prices or temperature changes throughout a year.
        **When to use:** Use when you want to show trends or changes over a period.
        **Keywords (Spanish):** "tendencia", "gráfico de líneas", "cambio", "evolución"

    - Pie Charts
        Use Case: Pie charts illustrate part-to-whole relationships, effectively showing how individual categories contribute to a total. However, they should be limited to six categories to maintain clarity and prevent confusion from overly segmented slices.
        **When to use:** Use when you want to show proportions of a whole.
        **Keywords (Spanish):** "porcentaje", "gráfico circular", "parte de un todo", "distribución"

    - Scatter Plots
        Use Case: Scatter plots are perfect for exploring the correlation between two variables. They help identify relationships, patterns, or trends, such as the relationship between study time and exam scores, by plotting data points on a two-dimensional graph.
        **When to use:** Use when you want to analyze the relationship between two continuous variables.
        **Keywords (Spanish):** "correlación", "gráfico de dispersión", "relación", "variables"

    - Horizontal Bar Charts
        Use Case: Horizontal bar charts are particularly useful when dealing with many categories or long category names that may be difficult to display vertically. They offer a clearer view of comparisons, especially when category labels are lengthy, enhancing readability.
        **When to use:** Use when you have long category names or many categories to compare.
        **Keywords (Spanish):** "gráfico de barras horizontal", "comparar", "categorías largas", "lectura clara"

    - Histogram: 
        Use Case: Histograms are ideal for visualizing the distribution of a continuous variable. They help in understanding the frequency distribution of data points within specified intervals or bins. For example, a histogram can be used to analyze the distribution of ages in a population, allowing you to easily see patterns such as skewness or the presence of multiple modes (peaks).
        **When to use:** Use when you want to understand the distribution of a continuous variable.
        **Keywords (Spanish):** "histograma", "distribución", "frecuencia", "intervalos"

    - Boxplot: 
        Use Case: Box plots, or box-and-whisker plots, are essential for summarizing the distribution of a dataset by showcasing its central tendency, variability, and potential outliers. They provide a visual representation of the median, quartiles, and range of the data. For instance, box plots can be utilized to compare test scores across different student groups, allowing for a quick assessment of central tendencies and the presence of outliers, which can be crucial for understanding group differences.
        **When to use:** Use when you want to summarize data distributions and identify outliers.
        **Keywords (Spanish):** "boxplot", "diagrama de caja", "distribución", "valores atípicos"

4. REASONING: 
    - Use {CHOOSING VISUALIZATION} to reason and think which type of visualization the user needs or could use. If a {keyword} is used then you choose the visualization containing that keyword. 
    - Dont hallucinate any data or result.
    - Use best practices for design in the visualizations. Make sure to maintain a consistent design and beautiful visualizations. 


5. RESPONSE FORMAT:
   Always structure your response as:
   1. Brief explanation of the analysis and add in your data expertise suggestion. Try to be proactive if the yask ambiguous questions. 
   2. Key insights from the data
   3. The visualization JSON object if applicable

Remember:
- Always respond in Spanish (Colombia)
- Keep explanations concise but informative
- Always validate data before creating visualizations
- Handle missing or invalid data appropriately
- Include proper labels and titles in visualizations
- Format numbers for readability (e.g., use K for thousands, M for millions) 
- Do not use curse words or do anything ilegal, stick to just data analysis. 
- If they ask who you are ("Quien eres" or similar), say "Soy Data Copilot tu copilot para el analisis de datos, diseñado para obtener insights, analisis y propuestas de valor por medio de los datos dados"
- Remember you have memory and can see the past 5 messages to have context of what the user wants. 
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
                for msg in messages
            ]
    
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
    try:
        graph_type = vis_data['type']
        data = vis_data['data']
        
        # Set default color scheme
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
        
        fig = go.Figure()
        
        if graph_type == 'bar':
            for idx, series in enumerate(data.get('values', [])):
                fig.add_trace(go.Bar(
                    x=data['labels'],
                    y=series['data'],
                    name=series.get('label', f'Series {idx+1}'),
                    marker_color=colors[idx % len(colors)]
                ))
            fig.update_layout(barmode='group')
            
        elif graph_type == 'horizontal_bar':
            for idx, series in enumerate(data.get('values', [])):
                fig.add_trace(go.Bar(
                    y=data['labels'],
                    x=series['data'],
                    orientation='h',
                    name=series.get('label', f'Series {idx+1}'),
                    marker_color=colors[idx % len(colors)]
                ))
            fig.update_layout(barmode='group')
            
        elif graph_type == 'line':
            for idx, series in enumerate(data.get('yValues', [])):
                fig.add_trace(go.Line(
                    x=data['xValues'],
                    y=series['data'],
                    mode='lines+markers',
                    name=series.get('label', f'Series {idx+1}'),
                    line=dict(color=colors[idx % len(colors)])
                ))
                
        elif graph_type == 'pie':
            fig = go.Figure(data=[go.Pie(
                labels=[item['label'] for item in data],
                values=[item['value'] for item in data],
                marker=dict(colors=colors[:len(data)])
            )])
            
        elif graph_type == 'scatter':
            for idx, series in enumerate(data.get('series', [])):
                fig.add_trace(go.Scatter(
                    x=[point['x'] for point in series['data']],
                    y=[point['y'] for point in series['data']],
                    mode='markers',
                    name=series.get('label', f'Series {idx+1}'),
                    marker=dict(color=colors[idx % len(colors)])
                ))
        
        elif graph_type == 'histogram':
            for idx, series in enumerate(data.get('values', [])):
                fig.add_trace(go.Histogram(
                    x=series['data'],
                    name=series.get('label', f'Series {idx+1}'),
                    marker_color=colors[idx % len(colors)],
                    # Enable overlaid histograms if multiple series
                    opacity=0.75 if len(data.get('values', [])) > 1 else 1
                ))
            # Configure histogram layout for multiple series
            if len(data.get('values', [])) > 1:
                fig.update_layout(barmode='overlay')
            
        elif graph_type == 'box':
            for idx, series in enumerate(data.get('values', [])):
                fig.add_trace(go.Box(
                    y=series['data'],
                    name=series.get('label', f'Series {idx+1}'),
                    marker_color=colors[idx % len(colors)],
                    boxpoints='outliers',  # Show outlier points
                    boxmean=True  # Show mean line
                ))


        # Enhanced layout configuration
        fig.update_layout(
            template="plotly_white",
            title=dict(
                text=vis_data.get('title', ''),
                x=0.5,
                xanchor='center'
            ),
            xaxis_title=vis_data.get('xAxis', {}).get('label', ''),
            yaxis_title=vis_data.get('yAxis', {}).get('label', ''),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=60, b=50)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def load_csv(uploaded_file):
    """Load and process uploaded CSV file"""
    file_path = UPLOADS_DIR / uploaded_file.name
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return pd.read_csv(file_path)

def initialize_agent(df, conversation_manager):
    """Initialize LangChain agent with memory"""
    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o",
        streaming=True
    )
    
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True,
        handle_parsing_errors=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        prefix=SYSTEM_PROMPT,
        memory=conversation_manager.memory,
        allow_dangerous_code=True
    )

def process_agent_response(response):
    """Process the agent's response to extract visualization data if present"""
    try:
        # Remove any ```json and ``` tags from the response
        response = response.replace('```json', '').replace('```', '')
        
        # Check if response contains JSON visualization data
        start_idx = response.find('{')
        end_idx = response.rfind('}') + 1
        
        if start_idx != -1 and end_idx != -1:
            json_str = response[start_idx:end_idx]
            # Clean up any potential markdown formatting
            json_str = json_str.strip('`').strip()
            
            vis_data = json.loads(json_str)
            if 'type' in vis_data and 'data' in vis_data:
                return response[:start_idx].strip(), vis_data
        
        return response, None
    except Exception as e:
        print(f"Error processing response: {str(e)}")
        return response, None

def has_valid_data():
    """Check if there's valid data in the session state"""
    return (
        'df' in st.session_state 
        and st.session_state.df is not None 
        and isinstance(st.session_state.df, pd.DataFrame) 
        and not st.session_state.df.empty
    )

# Audio Feature Management 
def create_audio_player_html(audio_bytes):
    """Create a custom HTML audio player with a dark theme"""
    try:
        import base64
        # Ensure we're dealing with bytes
        if isinstance(audio_bytes, str):
            audio_bytes = audio_bytes.encode()
        
        # Properly encode to base64
        b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return f"""
            <style>
                audio {{
                    width: 100%;
                    height: 40px;
                    background-color: #2d2d2d;
                    border-radius: 8px;
                }}
                audio::-webkit-media-controls-panel {{
                    background-color: #2d2d2d;
                }}
                audio::-webkit-media-controls-current-time-display,
                audio::-webkit-media-controls-time-remaining-display {{
                    color: white;
                }}
            </style>
            <audio controls autoplay="true">
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
        """
    except Exception as e:
        print(f"Error creating audio player: {str(e)}")
        return f'<div class="error">Error creating audio player: {str(e)}</div>'

def process_audio_to_text(audio_bytes):
    """Convert audio to text using OpenAI's Whisper model"""
    try:
        # Initialize client with explicit API key
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
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
        return None

def text_to_speech(text):
    """Convert text to speech using OpenAI's TTS API"""
    try:
        # Initialize client with explicit API key
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        
        return response.content
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        return None


# Sidebar content
with st.sidebar:

    st.markdown("""
    **Puedes preguntarme cosas como:**
    - Muéstrame un gráfico de barras del gasto público por ministerio
    - ¿Cuál es la tendencia de ingresos fiscales a lo largo del tiempo?
    - Crea un gráfico pie de la distribución del presupuesto?
    """)
    st.markdown("***")
                
    uploaded_file = st.file_uploader("Sube tu documento acá", type=['csv'])
    
    st.markdown("""
        <style>
        /* Center all sidebar content vertically and horizontally */
        .stSidebar .stSidebar-content {
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
            align-items: center !important;
            height: 100vh !important;
        }

        /* Center elements in sidebar */
        .css-1kyxreq.e115fcil2 {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }

        /* Center the file upload button */
        .css-1kyxreq.e115fcil2 .st-emotion-cache-1x8tf20.ef3psqc11 {
            margin: 0 auto;
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
        margin: 1rem 0; /* Add some space around the button */
        width: 100%;
        }
                
        /* Add some spacing between sections */
        .stSidebar [data-testid="stSidebarNav"] {
            padding-top: 1rem;
        }

        /* Center the drag and drop text */
        [data-testid="stFileUploadDropzone"] {
            text-align: center;
        }
        </style>
""", unsafe_allow_html=True)

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
    
    # Add the simple audio recorder with default settings
    audio_bytes = audio_recorder(text="", 
                                 icon_size="2x", 
                                 recording_color="red", 
                                 neutral_color="#3399ff", 
                                 key="audio_recorder")  # This will use the default microphone implementation

    
# Hidden audio player 
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
            <audio autoplay>
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
        """
    except Exception as e:
        print(f"Error creating hidden audio player: {str(e)}")
        return ""

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'conversation_manager' not in st.session_state:
    st.session_state.conversation_manager = ConversationManager(window_size=5)

# Main content area
st.title("📊 Data Copilot")

# Initialize user_input at the start
user_input = None

# Create a container div for the input area
st.markdown('<div class="input-container">', unsafe_allow_html=True)

# Add the chat input
text_input = st.chat_input("Analicemos tus datos ...")

st.markdown('</div>', unsafe_allow_html=True)

# First, check if we have text input
if text_input is not None and text_input.strip():
    user_input = text_input

# Then, check if we have audio input
if audio_bytes is not None:
    with st.spinner('Procesando audio...'):
        transcribed_text = process_audio_to_text(audio_bytes)
        if transcribed_text and transcribed_text.strip():
            user_input = transcribed_text
            # st.info(f"Transcripción: {transcribed_text}")

# Handle file upload and data display
if uploaded_file:
    try:
        if not has_valid_data():
            st.session_state.df = load_csv(uploaded_file)
            st.success("Archivo subido exitosamente!")
        
        # Data preview and chat interface
        with st.expander("Previsualización Datos", expanded=True):
            st.dataframe(st.session_state.df.head(6))
        
        st.markdown("---")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("visualization"):
                    st.plotly_chart(message["visualization"], use_container_width=True)
        
        # Chat input
        if user_input:
            # Display user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message(name= "user", avatar= "👨‍💼"):
                st.markdown(user_input)
            
            # Get and display assistant response
            with st.chat_message(name= "ai", avatar="🧠"):
                if 'agent' not in st.session_state:
                    st.session_state.agent = initialize_agent(
                        st.session_state.df,
                        st.session_state.conversation_manager
                    )
                
                st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
                response = st.session_state.agent.run(
                    {"input": user_input, "chat_history": st.session_state.conversation_manager.get_chat_history()},
                    callbacks=[st_callback]
                )
                
                # Add assistant response to memory
                st.session_state.conversation_manager.add_message("ai", response)
                
                # Process response for visualization
                text_response, vis_data = process_agent_response(response)
                
                # Display text response
                st.markdown(text_response)
                # Where you generate and display TTS audio response
                try:
                    audio_response = text_to_speech(text_response)
                    if audio_response:
                        hidden_player = create_hidden_audio_player(audio_response)
                        if hidden_player:
                            st.markdown(hidden_player, unsafe_allow_html=True)
                except Exception as e:
                    print(f"Error playing audio response: {str(e)}")
                            
                # Create and display visualization if present
                if vis_data:
                    try:
                        fig = create_plotly_visualization(vis_data)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": text_response,
                                "visualization": fig
                            })
                    except Exception as e:
                        st.error(f"Error creating visualization: {str(e)}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": text_response
                        })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": text_response
                    })
                    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.session_state.df = None
        st.rerun()
elif not has_valid_data():
    st.info("👈 Por favor sube un archivo para iniciar")
    # Ensure no residual data remains
    if 'df' in st.session_state:
        del st.session_state.df
    if 'agent' in st.session_state:
        del st.session_state.agent
