from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)
new_db = FAISS.load_local("faiss_db", embedding_model, allow_dangerous_deserialization=True)
retriever=new_db.as_retriever()
retrieval_chain= create_retriever_tool(retriever,"pdf_extractor", "This tool is to give answer to queries")


def get_conversational_chain(tools,ques):
    #os.environ["ANTHROPIC_API_KEY"]=os.getenv["ANTHROPIC_API_KEY"]
    #llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"),verbose=True)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key="****")
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an efficient and discerning assistant. First, carefully evaluate all the retrieved information and filter out anything that is not directly relevant to the user's question. Then, summarize the relevant information, and present your response in the format specified by the user. If no relevant answer is found in the provided context, respond with 'The answer is not available in the context.' Your response should be clear, concise, and directly aligned with the user's request, ensuring that the final output is well-organized and meets the desired format.""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    tool=[tools]
    agent = create_tool_calling_agent(llm, tool, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    response=agent_executor.invoke({"input": ques})
    return response

def process_user_input(user_question):
    return get_conversational_chain(retrieval_chain, user_question)

if __name__ == "__main__":
    user_question = "In canada what are the packaging rules for cannabies?"
    process_user_input(user_question)
