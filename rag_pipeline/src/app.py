import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.base.llm_model import get_hf_llm
from src.rag.main import build_rag_chain, InputQA, OutputQA


app = FastAPI(
    title="Langchain Server",
    version="1.0.0",
    description="A simple RAG API server using Langchain"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

genai_chain = None

@app.get("/")
async def root():
    return {"message": "RAG Pipeline Server", "status": "running"}


@app.get("/health")
async def health_check():
    global genai_chain
    if genai_chain is None:
        return {"status": "error", "message": "RAG chain not available"}
    return {"status": "ok", "message": "RAG chain available"}

@app.post("/rag/query", response_model=OutputQA)
async def rag_query(inputs: InputQA):
    global genai_chain
    if genai_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not available")
    
    try:
        result = genai_chain.invoke(inputs.question)
        
        if isinstance(result, dict):
            if "answer" in result:
                answer = result["answer"]
            elif "text" in result:
                answer = result["text"]
            elif "output" in result:
                answer = result["output"]
            else:
                answer = str(result)
        else:
            answer = str(result)
        
        return {"answer": answer}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize RAG chain on startup"""
    global genai_chain
    try:
        
        llm = get_hf_llm(
            model_name="microsoft/phi-2",  
            temperature=0.9,
            max_new_token=512  
        )
        
        genai_docs = "./data_source/generative_ai"
        genai_chain = build_rag_chain(llm, data_dir=genai_docs, data_type="pdf")
                
    except Exception as e:
        genai_chain = None
