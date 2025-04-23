from fastapi import FastAPI, Request
import uvicorn
import asyncio
from contextlib import asynccontextmanager

try:
    import sglang as sgl
except ImportError as e:
    pass

engine = None


# Use FastAPI's lifespan manager to initialize/shutdown the engine
@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    # Initialize the SGLang engine when the server starts
    # Adjust model_path and other engine arguments as needed
    print("Loading SGLang engine...")
    engine = sgl.Engine(model_path="Qwen/Qwen2-7B-Instruct", tp_size=1)
    print("SGLang engine loaded.")
    yield
    # Clean up engine resources when the server stops (optional, depends on engine needs)
    print("Shutting down SGLang engine...")
    # engine.shutdown() # Or other cleanup if available/necessary
    print("SGLang engine shutdown.")


app = FastAPI(lifespan=lifespan)


@app.post("/generate")
async def generate_text(request: Request):
    global engine
    if not engine:
        return {"error": "Engine not initialized"}, 503

    try:
        data = await request.json()
        prompt = data.get("prompt")
        max_new_tokens = data.get("max_new_tokens", 128)
        temperature = data.get("temperature", 0.7)

        if not prompt:
            return {"error": "Prompt is required"}, 400

        # Use async_generate for non-blocking generation
        state = await engine.async_generate(
            prompt,
            sampling_params={
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
            # Add other parameters like stop, top_p etc. as needed
        )

        return {"generated_text": state["text"]}
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == "__main__":
    # Required for async_generate if running in a nested loop environment like Jupyter
    # import nest_asyncio
    # nest_asyncio.apply()

    uvicorn.run(app, host="0.0.0.0", port=8000)
