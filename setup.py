from setuptools import find_packages, setup

setup(
    name="ecommbot",
    version="0.0.1",
    author="Shaurab",
    author_email="shaurbaio@gmail.com",
    description="E-commerce chatbot using LangChain, AstraDB, and Mistral 7B via Hugging Face.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "langchain",
        "langchain-core",
        "langchain-community",
        "langchain-astradb",
        "datasets",
        "pypdf",
        "python-dotenv",
        "flask",
        "requests",
        "transformers",
        "huggingface_hub"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
