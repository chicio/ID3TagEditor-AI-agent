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

    const model = new ChatOllama({
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

    const chain = await createStuffDocumentsChain({
        llm: model,
        prompt,
        outputParser: new StringOutputParser(),
    });

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });

    console.log('Chat started... Use /stop for exit');

    const askQuestion = () => {
        rl.question('Question: ', async (question) => {
            if (question.toLowerCase() === '/stop') {
                console.log('Good bye!!');
                rl.close();
                return;
            }

            try {
                console.log('\n-------------------');
                const docsWithScores = await vectorStore.similaritySearchWithScore(question, 10);
                
                const threshold = 0.7;
                const filteredDocs = docsWithScores
                    .filter(([_, score]) => score >= threshold)
                    .map(([doc, score]) => {
                        doc.metadata.score = score;
                        return doc;
                    });

                const response = await chain.invoke({
                    question,
                    context: filteredDocs,
                });

                console.log('\nAnswer:', response);
                console.log('\n-------------------\n');
                
                askQuestion();
            } catch (error) {
                console.error('❌ Errore:', error);
                askQuestion();
            }
        });
    };

    askQuestion();
};

runChat().catch(error => {
    console.error('❌ Fatal error:', error);
    process.exit(1);
}); 