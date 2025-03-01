import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OllamaEmbeddings } from "@langchain/ollama";
import { ChatOllama } from "@langchain/ollama";
import { ChatPromptTemplate, PromptTemplate } from "@langchain/core/prompts";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { Document } from "@langchain/core/documents";

export const chatModel = async () => {    
    const embeddings = new OllamaEmbeddings({
        model: "codellama:7b", 
        keepAlive: "30m",
    });
    const llmModel = new ChatOllama({
        model: "codellama:7b", 
        keepAlive: "30m",
        streaming: true,
    });
    const prompt = PromptTemplate.fromTemplate(`
        You're an expert programmer. Use the information to answer question about the ID3TagEditor codebase, how I can improve it and 
        useful information about it. Answer 'I don't know' if there is no useful information in the context.
        
        Context: {context}
        Question: {question}
        
        Answer:`);    

    const vectorStore = await FaissStore.load("faiss_index", embeddings);

    const InputStateAnnotation = Annotation.Root({
        question: Annotation<string>,
    });
  
    const StateAnnotation = Annotation.Root({
        question: Annotation<string>,
        context: Annotation<Document[]>,
        answer: Annotation<string>,
    });
  
    const retrieve = async (state: typeof InputStateAnnotation.State) => {
        const retrievedDocs = await vectorStore.similaritySearchWithScore(state.question, 30)
        const threshold = 0.;
        const filteredDocs = retrievedDocs
            .filter(([_, score]) => score >= threshold)
            .map(([doc, score]) => {
                doc.metadata.score = score;
                return doc;
            });
        return { context: filteredDocs };
    };
    
    const generate = async (state: typeof StateAnnotation.State) => {
        const docsContent = state.context.map(doc => doc.pageContent).join("\n");
        const messages = await prompt.invoke({ question: state.question, context: docsContent });
        const response = await llmModel.invoke(messages);
        return { answer: response.content };
    };
  
    const graph = new StateGraph(StateAnnotation)
        .addNode("retrieve", retrieve)
        .addNode("generate", generate)
        .addEdge("__start__", "retrieve")
        .addEdge("retrieve", "generate")
        .addEdge("generate", "__end__")
        .compile();

    return graph        
};