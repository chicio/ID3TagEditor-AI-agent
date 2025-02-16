import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OllamaEmbeddings } from "@langchain/ollama";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatOllama } from "@langchain/ollama";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { PromptTemplate } from "@langchain/core/prompts";
import * as readline from 'readline';

const runChat = async () => {
    console.log('Caricamento indice FAISS...');
    const embeddings = new OllamaEmbeddings({
        model: "codellama:7b",
        keepAlive: "30m",
    });
    const vectorStore = await FaissStore.load("faiss_index", embeddings);

    // Configura il modello di chat
    const model = new ChatOllama({
        model: "codellama:7b",
        keepAlive: "30m",
        streaming: true,
    });

    // Crea il prompt template per il RAG
    const prompt = PromptTemplate.fromTemplate(`
    You're an expert programmer. Use the information to answer question about the ID3TagEditor codebase, how I can improve it and 
    useful information about it.
    
    Context: {context}
    Question: {question}
    
    Answer:`);

    // Crea la catena RAG
    const chain = await createStuffDocumentsChain({
        llm: model,
        prompt,
        outputParser: new StringOutputParser(),
    });

    // Configura l'interfaccia readline
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    console.log('Chat started... Use /stop for exit');

    // Loop principale della chat
    const askQuestion = () => {
        rl.question('Question: ', async (question) => {
            if (question.toLowerCase() === '/stop') {
                console.log('Good bye!!');
                rl.close();
                return;
            }

            try {
                // Recupera i documenti rilevanti
                const docs = await vectorStore.similaritySearch(question, 10);

                // Esegue la catena RAG
                const response = await chain.invoke({
                    question,
                    context: docs,
                });

                console.log('\nAnswer:', response);
                console.log('\n-------------------\n');
                
                askQuestion();
            } catch (error) {
                console.error('Errore:', error);
                askQuestion();
            }
        });
    };

    askQuestion();
};

runChat().catch(console.error); 