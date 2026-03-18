from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import uvicorn

from inference import verify_sig
from loader import load_signet_model

app = FastAPI()

model = None
device = None

@app.on_event("startup")
async def startup_event():
    global model, device
    model, device = load_signet_model('SigNet.pt')
    model.eval()


@app.post("/verify-sig")
async def verify_signature(file1: UploadFile = File(...),file2:UploadFile = File(...),threshold:float = 0.5)->dict:
    image1_bytes = await file1.read()
    image2_bytes = await file2.read()

    img1_arr = np.frombuffer(image1_bytes, np.uint8)
    img2_arr = np.frombuffer(image2_bytes, np.uint8)

    image1 = cv2.imdecode(img1_arr, cv2.IMREAD_COLOR)
    image2 = cv2.imdecode(img2_arr,cv2.IMREAD_COLOR)

    if image1 is None:
        return JSONResponse(
            status_code=400, 
            content={"error": "Invalid image format provided."}
        )
    if image2 is None:
        return JSONResponse(
            status_code=400, 
            content={"error": "Invalid image format provided."}
        )


    verification = verify_sig(image1,image2,model,device,threshold)


    return verification

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)