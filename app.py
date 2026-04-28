import sys
import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import torch
from PIL import Image
from torchvision import transforms
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash

# =======================
# PATHS & IMPORTS
# =======================

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

from src.model import CNNModel
from src.gradcam import generate_heatmap

# =======================
# FLASK APP
# =======================

app = Flask(__name__)
CORS(app, origins=[os.environ.get('FRONTEND_URL', '*')])

# FIX: Use PostgreSQL on Render (falls back to SQLite for local dev)
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///users.db')
# Render PostgreSQL URLs start with postgres://, SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'pancreas-cancer-detection-secret-key')

db = SQLAlchemy(app)
jwt = JWTManager(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

with app.app_context():
    db.create_all()

UPLOAD_FOLDER = os.path.join(ROOT_DIR, "uploads")
STATIC_FOLDER = os.path.join(ROOT_DIR, "static")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# =======================
# MODEL LOADING (only once here)
# =======================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNNModel(num_classes=2).to(device)
model.load_state_dict(
    torch.load(
        os.path.join(ROOT_DIR, "saved_models/pancreas_model.pth"),
        map_location=device
    )
)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =======================
# API ROUTES
# =======================

@app.route("/api/auth/register", methods=["POST"])
def register():
    data = request.get_json()
    if not data or not data.get("email") or not data.get("password") or not data.get("name"):
        return jsonify({"error": "Missing required fields"}), 400

    if User.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "User already exists"}), 400

    hashed_password = generate_password_hash(data["password"])
    new_user = User(name=data["name"], email=data["email"], password_hash=hashed_password)

    try:
        db.session.add(new_user)
        db.session.commit()
        return jsonify({"message": "User registered successfully"}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Registration failed"}), 500


@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    if not data or not data.get("email") or not data.get("password"):
        return jsonify({"error": "Missing required fields"}), 400

    user = User.query.filter_by(email=data["email"]).first()
    if not user or not check_password_hash(user.password_hash, data["password"]):
        return jsonify({"error": "Invalid email or password"}), 401

    access_token = create_access_token(identity=str(user.id))
    return jsonify({
        "message": "Login successful",
        "access_token": access_token,
        "user": {"name": user.name, "email": user.email}
    }), 200


@app.route("/")
def home():
    return jsonify({"message": "Pancreas Cancer Detection API Running"})


@app.route("/predict", methods=["POST"])
@jwt_required()
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)

    try:
        # Image preprocessing
        img = Image.open(img_path).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(x)
            probabilities = torch.nn.functional.softmax(out, dim=1)
            cancer_prob = probabilities[0][0].item()
            normal_prob = probabilities[0][1].item()

            # Class 0 = Cancer, Class 1 = Normal
            if cancer_prob > 0.50:
                prediction = "Cancer"
                confidence_val = cancer_prob
            else:
                prediction = "Normal"
                confidence_val = normal_prob

        # FIX: Pass model and device into generate_heatmap (no double loading)
        heatmap_path = generate_heatmap(img_path, model, device)

        # Move heatmap to static folder
        final_heatmap_path = os.path.join(STATIC_FOLDER, "heatmap.jpg")
        os.replace(heatmap_path, final_heatmap_path)

        # FIX: Return full URL so frontend (Vercel) can access it
        base_url = os.environ.get('RENDER_EXTERNAL_URL', '')
        heatmap_url = f"{base_url}/static/heatmap.jpg"

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence_val * 100, 2),
            "heatmap_url": heatmap_url
        })

    finally:
        # FIX: Clean up uploaded file after prediction to save disk space
        if os.path.exists(img_path):
            os.remove(img_path)


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)


# =======================
# MAIN
# =======================

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
