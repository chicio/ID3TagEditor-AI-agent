# ID3TagEditor-AI-Agent

The **ID3TagEditor-AI-Agent** is an AI-powered chatbot that allows users to query the [**ID3TagEditor**](https://github.com/chicio/ID3TagEditor) codebase using Retrieval-Augmented Generation (RAG) techniques. It leverages LangChain, Ollama, and FAISS to retrieve relevant information from the codebase and provide answers to questions about the project.

## Features

- **RAG-Driven Chatbot**: The chatbot uses RAG to answer questions related to the **ID3TagEditor** codebase, assisting with understanding the code and improving development processes.
- **Codebase Querying**: Users can ask questions like "How does this function work?" or "Where is this variable used?".
- **FAISS-Based Similarity Search**: Retrieves and uses relevant documents from the codebase to generate responses.
- **Ollama Integration**: Powered by the Ollama AI model for natural language processing and conversation.

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/chicio/ID3TagEditor-AI-agent.git
cd ID3TagEditor-AI-agent
npm install
```

## Usage

To interact with the chatbot, use the following commands:

- `npm run faiss`: Runs the FAISS index generation script.
- `npm run chat`: Starts the chatbot that allows you to query the codebase using RAG.

## Contributing

Feel free to fork the repository, open issues, and submit pull requests to improve the chatbot's functionality or add new features!

## License

MIT License.
