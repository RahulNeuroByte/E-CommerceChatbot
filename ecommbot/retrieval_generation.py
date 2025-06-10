import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFaceHub  # Correct HuggingFaceHub import
from ecommbot.ingest import ingestdata

def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    PRODUCT_BOT_TEMPLATE = """
    You are a smart product recommendation bot.
    Based on product reviews and titles, give a short and direct answer with a brief explanation.

    CONTEXT:
    {context}

    QUESTION: {question}

    ANSWER:
    """

    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)

    # Using HuggingFaceHub for model integration
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        huggingfacehub_api_token="your_huggingface_api_token_here",  # Replace with your token
        model_kwargs={
            "temperature": 0.7,
            "max_new_tokens": 200
        }
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

if __name__ == '__main__':
    vstore = ingestdata("done")  # Assuming this function adds data to your vector store
    chain = generation(vstore)
    answer = chain.invoke("Can you tell me the best Bluetooth buds?")
    print(answer.strip())
