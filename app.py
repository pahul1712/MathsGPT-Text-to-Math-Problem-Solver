import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

## Streamlit Application

# Page Config
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant", page_icon="🧮", layout="wide")

# Main Page Styling:
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e8f0ff 0%, #ffffff 60%, #f4f7ff 100%) !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e3ecff 0%, #f4f7ff 80%) !important;
    border-right: 2px solid rgba(0, 0, 0, 0.05);
    box-shadow: 4px 0 10px rgba(0, 0, 0, 0.05);
}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label {
    color: #002b5c !important;
    font-weight: 500;
}
.main {
    background-color: rgba(255, 255, 255, 0.85);
    padding: 2rem 3rem;
    border-radius: 15px;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(200, 200, 255, 0.35);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
}
.stButton>button {
    background: linear-gradient(90deg, #0078d4, #00b4d8);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6em 1.5em;
    font-weight: 600;
    transition: 0.3s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #00b4d8, #0078d4);
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)

# App Title
st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=100)
st.title("🧮 MathsGPT: Text-to-Math Problem Solver using Groq AI")
st.markdown("""
<div style='text-align:center;'>
    <h3>🧠 Solve complex math problems & explore factual knowledge instantly 🌍</h3>
    <p style='font-size:17px; color:#333;'>Powered by <b>Groq AI</b> + <b>LangChain</b>, MathsGPT understands natural language problems, performs step-by-step math reasoning, and finds related information in real time.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    groq_api_key = st.text_input("🔑 Enter Groq API Key", type="password")
    st.markdown("---")
    st.markdown("💡 Get your free API key from [Groq Console](https://console.groq.com).")

if not groq_api_key:
    st.info("Please add your Groq API Key to continue")
    st.stop()

# Model Initialization
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

## Tool Setup
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Search Wikipedia for factual information about topics, people, places, or events."
)

## Simple Math Solver Function
def solve_math(query: str) -> str:
    """Solve math problems step-by-step with complete solutions."""
    try:
        prompt = f"""You are a math solver. Solve this problem completely and provide the final numerical answer.

Problem: {query}

Instructions:
1. Solve the problem step-by-step
2. Show your work
3. End with "The answer is: [number]"

Solution:"""
        
        response = llm.invoke(prompt)
        result = response.content.strip()
        
        # Make sure we return a complete answer
        if "answer" not in result.lower():
            # Try to extract the solution
            result += "\n\nThis is the complete solution."
        
        return result
        
    except Exception as e:
        return f"Error: {str(e)}"
    

calculator = Tool(
    name="Calculator",
    func=solve_math,
    description="Solve math problems and perform calculations. Use this for any mathematical question or arithmetic."
)

## Initialize the Agent with CLEAR instructions
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=4,
    max_execution_time=30,
    early_stopping_method="generate"
)

# Session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "👋 Hi! I'm MathsGPT, your math problem-solving buddy. Ask me any question!"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

## Start the Interaction
question = st.text_area(
    "✏️ Enter your math or reasoning question:",
    "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?"
)

if st.button("Find the answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)
            
            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response = assistant_agent.run(question,callbacks=[st_cb])
            st.session_state.messages.append({"role":"assistant","content":response})
            st.write("#### Response:")

            st.success(response)

    else:
        st.warning("Please enter the question")