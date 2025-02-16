import * as path from "path";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OllamaEmbeddings } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter, SupportedTextSplitterLanguages } from "@langchain/textsplitters";
import { error } from "console";

// Funzione per caricare la codebase
const loadCodebase = () => {
  const loader = new DirectoryLoader(path.join(__dirname, "../ID3TagEditor/Source"), {
    ".swift": (filePath: string) => new TextLoader(filePath)
  });
  return loader.load();
};

// Funzione per creare l'indice FAISS
const createIndex = async () => {
  console.log('starting load codebase...')
  const documents = await loadCodebase();
  console.log('codebase loaded...')

  console.log('starting split...')
  console.log(SupportedTextSplitterLanguages);
  const splitter = RecursiveCharacterTextSplitter.fromLanguage("swift", {
    chunkSize: 512, chunkOverlap: 200
  });

  const splitDocuments = await splitter.splitDocuments(documents)
  console.log('split completed...')

  console.log('starting FAISS index creation...')
  const embeddings = new OllamaEmbeddings({ model: "codellama:7b" });
  const vectorStore = new FaissStore(embeddings, {});

  for (let i = 0; i < documents.length - 1; i++) {
    console.log('Adding ' + documents[i].metadata.source)
    try {
      await vectorStore.addDocuments([documents[i]])
    } catch(error) {
      console.log(error)
    }
  }


  // const faissIndex = await FaissStore.fromDocuments(splitDocuments, embeddings);
  await vectorStore.save("faiss_index");

  console.log("FAISS Index created and saved.");
};

// Esegui la creazione dell'indice
createIndex().catch(console.error);
