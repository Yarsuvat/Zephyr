import time
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Zephyr Lite Core Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

class GenerationResponse(BaseModel):
    response: str
    generation_time: float
    device_used: str

class ZephyrLiteEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = None
        self.model = None

    def load_model(self):
        print(f"[*] Zephyr Lite yükleniyor... Cihaz: {self.device.upper()}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )

        if self.device == "cpu":
            self.model.to("cpu")

        print("[+] Zephyr Lite başarıyla belleğe yüklendi 🗿")

    def generate(self, prompt: str, max_tokens: int, temperature: float, top_p: float) -> str:
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model henüz yüklenmedi!")

        messages = [
            {"role": "system", "content": "Sen Zephyr Lite adında yardımcı bir yapay zekasın. Kullanıcıya Türkçe ve mantıklı yanıtlar ver."},
            {"role": "user", "content": prompt}
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response_tokens = outputs[0][inputs.input_ids.shape[1]:]
        response_text = self.tokenizer.decode(response_tokens, skip_special_tokens=True)

        return response_text.strip()

zephyr = ZephyrLiteEngine()

@app.on_event("startup")
async def startup_event():
    zephyr.load_model()

@app.get("/")
async def root():
    return {
        "status": "online",
        "system": "Zephyr Lite AI Engine",
        "device": zephyr.device
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "gpu_available": torch.cuda.is_available()}

@app.post("/api/v1/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    try:
        start_time = time.time()

        output_text = zephyr.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )

        elapsed_time = round(time.time() - start_time, 3)

        return GenerationResponse(
            response=output_text,
            generation_time=elapsed_time,
            device_used=zephyr.device
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)