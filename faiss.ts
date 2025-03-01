import * as path from "path";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { OllamaEmbeddings } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter, SupportedTextSplitterLanguages } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";

const loadCodebase = async (): Promise<Document[]> => {
  const loader = new DirectoryLoader(path.join(__dirname, "../ID3TagEditor/Source"), {
    ".swift": (filePath: string) => new TextLoader(filePath)
  });

  return loader.load();
};

const splitDocuments = async (documents: Document[]): Promise<Document[]>  => {
  const splitter = RecursiveCharacterTextSplitter.fromLanguage("swift", {
    chunkSize: 1024, chunkOverlap: 128
  });

  return await splitter.splitDocuments(documents)
}

const groupDocumentByPaths = (documents: Document[]): Map<string, Document[]> => {
  const splitGroupedByPath = new Map<string, Document[]>()

  documents.forEach(document => {
    let currentPathGroup: Document[] = []
    
    if (splitGroupedByPath.has(document.metadata.source)) {
      currentPathGroup = splitGroupedByPath.get(document.metadata.source)!
    }

    currentPathGroup.push(document)

    splitGroupedByPath.set(
      document.metadata.source, 
      currentPathGroup
    )
  })

  return splitGroupedByPath
}

const createIndex = async () => {
  const documents = await loadCodebase();
  const splittedDocuments = await splitDocuments(documents)
  const groupedDocuments = groupDocumentByPaths(splittedDocuments)

  const embeddings = new OllamaEmbeddings({ 
    model: "codellama:7b", 
    keepAlive: "30m",
  });
  const vectorStore = new FaissStore(embeddings, {});

  for await (const [path, documents] of groupedDocuments) {
    try {
      await vectorStore.addDocuments(documents)
      console.log(`✅ Added ${documents.length} chunks from ${path}`)
    } catch(error) {
      console.log(`❌ Error processing ${path}:`, error)
    }
  }

  await vectorStore.save("faiss_index");
};

// Esegui la creazione dell'indice
console.log("Starting FAISS Index creation.");
createIndex().catch(console.error);
console.log("FAISS Index created and saved.");
