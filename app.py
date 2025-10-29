import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler

## Streamlit Application
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assitant",page_icon="🧮")
st.title("🧮 MathsGPT: Text-to-Math Problem Solver using Meta Llama 3.3")

groq_api_key = st.sidebar.text_input("Groq API Key",type="password")


if not groq_api_key:
    st.info("Please add your Groq API Key to continue")
    st.stop()

llm = ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=groq_api_key)

## Initializing the Tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name = "Wikipedia",
    func = wikipedia_wrapper.run,
    description="A tool for searching the Internet to find the various information on the topics mentioned"
)

## Initialize the Math Tool

math_chain = LLMMathChain.from_llm(llm=llm)
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions. Only input mathematical expression need to be provided"
)

prompt="""
You are a step-by-step math reasoning assistant. 
When solving a question:
1. Break the problem into small steps.
2. Show the reasoning for each step clearly and logically.
3. Then provide the final answer at the end on a new line.

Question: {question}

Let's begin solving it step by step:
"""





prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)



## Combine all the tools into chain
chain = LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="A tool for answering logic-based and reasoning questions."
)


## Initialize the Agents

assistant_agent = initialize_agent(
    tools=[wikipedia_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant","content":"Hi, I'm a Math CHATBot who can answer all your math problems"}
    ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])


## Start the Interaction

question = st.text_area("Enter your question:","I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")

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

