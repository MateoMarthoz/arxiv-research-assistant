import threading
import uvicorn
import time
import gradio as gr
from app import main as fastapi_app 
import gradio_app       
def run_fastapi():
    # This runs your FastAPI app on a given host and port.
    uvicorn.run(fastapi_app.app, host="0.0.0.0", port=8000)

def run_gradio():
    # This launches the Gradio interface
    gradio_app.main()

if __name__ == "__main__":
    # Start FastAPI in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    
    #wait a bit for FastAPI to start (e.g. 2 seconds)
    time.sleep(2)
    
    # Launch the Gradio interface (this will block)
    run_gradio()
