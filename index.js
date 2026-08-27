import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import { MongoClient } from 'mongodb';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { TaskType } from "@google/generative-ai";
import { Embeddings } from '@langchain/core/embeddings';
import { MongoDBAtlasVectorSearch } from '@langchain/mongodb';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnablePassthrough, RunnableSequence } from '@langchain/core/runnables';

const MONGO_URI = process.env.MONGO_URI;
const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;
const DB_NAME = "gita_db";
const COLLECTION_NAME = "verses";
const INDEX_NAME = "gita_vector_index";
const EMBEDDING_MODEL = "gemini-embedding-001";
const EMBEDDING_DIMENSIONS = 1536;
const CHAT_MODEL = "gemini-3.6-flash";
const TOP_K = 3;

const app = express();
app.use(express.json());

app.use(cors({
  origin: [
    "http://localhost:3000",
    "https://bhagavad-gita.vercel.app"
  ],
  credentials: true
}));

// @langchain/google-genai's GoogleGenerativeAIEmbeddings does NOT support
// outputDimensionality — it silently ignores it and always requests the
// model's native 3072 dims. This wrapper calls @google/generative-ai
// directly, which does honor it, and matches the LangChain Embeddings interface
// so it drops straight into MongoDBAtlasVectorSearch.
class GeminiEmbeddings extends Embeddings {
  constructor(fields) {
    super(fields ?? {});
    this.model = fields.model ?? EMBEDDING_MODEL;
    this.taskType = fields.taskType;
    this.outputDimensionality = fields.outputDimensionality ?? EMBEDDING_DIMENSIONS;
    this.client = new GoogleGenerativeAI(GOOGLE_API_KEY)
      .getGenerativeModel({ model: this.model });
  }

  async _embed(text) {
    const res = await this.client.embedContent({
      content: { role: 'user', parts: [{ text }] },
      taskType: this.taskType,
      outputDimensionality: this.outputDimensionality,
    });
    return res.embedding.values;
  }

  async embedQuery(text) {
    return this._embed(text);
  }

  async embedDocuments(documents) {
    return Promise.all(documents.map((doc) => this._embed(doc)));
  }
}

let mongoClient;
let retriever;
let llm;

// Initialization function
const initializeApp = async () => {
  if (!MONGO_URI) {
    throw new Error("MONGO_URI is missing from environment variables.");
  }
  if (!GOOGLE_API_KEY) {
    throw new Error("GOOGLE_API_KEY is missing from environment variables.");
  }

  mongoClient = new MongoClient(MONGO_URI);
  await mongoClient.connect();

  const collection = mongoClient.db(DB_NAME).collection(COLLECTION_NAME);

  const embeddings = new GeminiEmbeddings({
    model: EMBEDDING_MODEL,
    taskType: TaskType.RETRIEVAL_QUERY, // queries use RETRIEVAL_QUERY, not RETRIEVAL_DOCUMENT
    outputDimensionality: EMBEDDING_DIMENSIONS,
  });

  const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
    collection,
    indexName: INDEX_NAME,
    textKey: "text",
    embeddingKey: "embedding"
  });

  retriever = vectorStore.asRetriever({
    k: TOP_K
  });

  llm = new ChatGoogleGenerativeAI({
    model: CHAT_MODEL,
    temperature: 0.2,
  });

  console.log("Connected to MongoDB and initialized LangChain components.");
};

// Prompt Template
const promptTemplate = ChatPromptTemplate.fromTemplate(
  `You are a guide answering questions about the Bhagavad Gita, using the verses below as your only source of teaching.

The question may not use the same words as the verses — connect the underlying theme of the verses to what's being asked, the way a teacher would. Weave the verses together into a single coherent answer; do not treat them as separate quotes to list. Reference the chapter and verse naturally inline (e.g. "as Krishna tells Arjuna in 2.3...").

Only say the verses don't address the question if they are genuinely unrelated in theme, not just in wording.

Verses:
{context}

Question: {question}

Answer:`
);

const formatDocs = (docs) =>
  docs
    .map(d => {
      const chapter = d.metadata?.chapter ?? '';
      const verse = d.metadata?.verse ?? '';
      return `[Ch ${chapter}.${verse}] ${d.pageContent}`;
    })
    .join('\n\n');

// Routes
app.post('/chat', async (req, res) => {
  const { query } = req.body;

  if (!query || typeof query !== 'string' || !query.trim()) {
    return res.status(400).json({ detail: "query is required" });
  }

  try {
    const docs = await retriever.invoke(query);
    const contextStr = formatDocs(docs);

    const chain = RunnableSequence.from([
      {
        context: () => contextStr,
        question: new RunnablePassthrough()
      },
      promptTemplate,
      llm,
      new StringOutputParser()
    ]);

    const answer = await chain.invoke(query);

    const sources = docs.map(d => ({
      text: d.pageContent,
      chapter: d.metadata?.chapter ?? null,
      verse: d.metadata?.verse ?? null,
      source: d.metadata?.source ?? null,
      sloka: d.metadata?.sloka ?? null
    }));

    return res.json({
      answer,
      sources
    });

  } catch (error) {
    console.error("Error during chat processing:", error);
    return res.status(500).json({ detail: error.message || "An internal error occurred" });
  }
});

app.get('/health', (req, res) => {
  return res.json({ status: "ok" });
});

// Startup and Shutdown
const PORT = process.env.PORT || 8000;

initializeApp()
  .then(() => {
    const server = app.listen(PORT, () => {
      console.log(`Server running on port ${PORT}`);
    });

    const gracefulShutdown = () => {
      console.log("Shutting down gracefully...");
      server.close(async () => {
        if (mongoClient) {
          await mongoClient.close();
          console.log("MongoDB connection closed.");
        }
        process.exit(0);
      });
    };

    process.on('SIGTERM', gracefulShutdown);
    process.on('SIGINT', gracefulShutdown);
  })
  .catch(err => {
    console.error("Failed to initialize system:", err);
    process.exit(1);
  });