from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from PIL import Image
import shutil
import uuid
import os
import logging
import torch
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
from ultralytics import YOLO
from fpdf import FPDF
import smtplib
from email.message import EmailMessage

app = FastAPI()

# Middleware
app.add_middleware(SessionMiddleware, secret_key="supersecret")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Folders
UPLOAD_DIR, RESULTS_DIR, STATIC_DIR, PDF_DIR, USER_DIR = "uploads", "results", "static", "pdfs", "users"
for d in [UPLOAD_DIR, RESULTS_DIR, STATIC_DIR, PDF_DIR, USER_DIR]:
    os.makedirs(d, exist_ok=True)

app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# OAuth Setup
config = Config(environ={
    "GOOGLE_CLIENT_ID": "your-client-id",
    "GOOGLE_CLIENT_SECRET": "your-client-secret"
})
oauth = OAuth(config)
google = oauth.register(
    name="google",
    client_id=os.getenv("GOOGLE_CLIENT_ID", "your-client-id"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET", "your-client-secret"),
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

# AI Models
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")
object_detector = YOLO("yolov8n.pt")

PRODUCT_DB = {
    "couch": {"name": "Modern Couch", "price": "$799", "link": "https://example.com/product/couch", "image": "/static/sample-product-couch.jpg"},
    "chair": {"name": "Accent Chair", "price": "$299", "link": "https://example.com/product/chair", "image": "/static/sample-product-chair.jpg"},
    "bed": {"name": "Queen Bed Frame", "price": "$599", "link": "https://example.com/product/bed", "image": "/static/sample-product-bed.jpg"},
    "tv": {"name": "Smart TV", "price": "$499", "link": "https://example.com/product/tv", "image": "/static/sample-product-tv.jpg"},
    "dining table": {"name": "Dining Table", "price": "$699", "link": "https://example.com/product/dining-table", "image": "/static/sample-product-table.jpg"}
}

@app.route("/login")
async def login(request: Request):
    redirect_uri = request.url_for("auth")
    return await google.authorize_redirect(request, redirect_uri)

@app.route("/auth")
async def auth(request: Request):
    token = await google.authorize_access_token(request)
    user = await google.parse_id_token(request, token)
    request.session["user"] = user
    return {"message": "Login successful", "user": user}

@app.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return {"message": "Logged out"}

@app.get("/me")
async def me(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

@app.post("/transform")
async def transform_room(file: UploadFile = File(...), vibe: str = Form(...), request: Request = None):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    init_image = Image.open(file_path).convert("RGB").resize((768, 512))
    result = pipe(prompt=vibe, image=init_image, strength=0.75, guidance_scale=7.5)
    transformed_image = result.images[0]
    transformed_path = os.path.join(RESULTS_DIR, f"transformed_{file_id}.png")
    transformed_image.save(transformed_path)

    results = object_detector(transformed_path)[0]
    labels = [object_detector.model.names[int(cls)] for cls in results.boxes.cls] if results.boxes.cls.numel() else []
    seen = set()
    products = [PRODUCT_DB[label] for label in labels if label in PRODUCT_DB and not seen.add(label)]

    user_dir = os.path.join(USER_DIR, user['email'].replace('@', '_'))
    os.makedirs(user_dir, exist_ok=True)
    shutil.copy(transformed_path, os.path.join(user_dir, f"transformed_{file_id}.png"))

    return {"transformed_image_url": f"/results/transformed_{file_id}.png", "products": products}

class EmailPDFRequest(BaseModel):
    image: str
    products: list
    email: str

@app.post("/email-pdf")
async def email_pdf(data: EmailPDFRequest):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Your AI Room Transformation", ln=True, align="C")
    img_path = "." + data.image
    if os.path.exists(img_path):
        pdf.image(img_path, x=10, y=20, w=180)
    pdf.ln(110)
    pdf.cell(200, 10, txt="Items Detected:", ln=True)
    for item in data.products:
        pdf.cell(200, 10, txt=f"{item['name']} - {item['price']} - {item['link']}", ln=True)

    pdf_name = f"{uuid.uuid4()}.pdf"
    pdf_path = os.path.join(PDF_DIR, pdf_name)
    pdf.output(pdf_path)

    msg = EmailMessage()
    msg["Subject"] = "Your AI Room Design Results"
    msg["From"] = "noreply@icl.co"
    msg["To"] = data.email
    msg.set_content("Attached is the PDF of your room transformation and products.")
    with open(pdf_path, "rb") as f:
        msg.add_attachment(f.read(), maintype="application", subtype="pdf", filename="Room_Design.pdf")
    with smtplib.SMTP("localhost") as smtp:
        smtp.send_message(msg)

    return {"status": "Email sent successfully."}

@app.get("/dashboard")
async def dashboard(request: Request):
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user_dir = os.path.join(USER_DIR, user['email'].replace('@', '_'))
    return {"designs": os.listdir(user_dir) if os.path.exists(user_dir) else []}
