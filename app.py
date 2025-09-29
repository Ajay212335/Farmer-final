from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from flask_session import Session
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

import os
import pickle
import numpy as np
import razorpay

# ----------------------------
# App & Environment Setup
# ----------------------------
app = Flask(__name__)
load_dotenv()  # Load .env if present

# Secrets / Config from environment (Render -> Dashboard -> Environment)
app.secret_key = os.getenv("SECRET_KEY", "change_me_in_env")
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
Session(app)

# MongoDB (Use Render/Atlas connection string via env)
app.config["MONGO_URI"] = os.getenv("MONGO_URI", "mongodb+srv://prasheetha:prasheetha37@cluster0.wne9ze2.mongodb.net/loginapp?retryWrites=true&w=majority&appName=Cluster0")
mongo = PyMongo(app)

# Google API
if os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Razorpay
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "YOUR_RAZORPAY_KEY_ID")
RAZORPAY_SECRET = os.getenv("RAZORPAY_SECRET", "YOUR_RAZORPAY_SECRET")
razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_SECRET))

# Ensure uploads folder exists
os.makedirs(os.path.join("static", "uploads"), exist_ok=True)

# ----------------------------
# Health Check (useful for Render)
# ----------------------------
@app.route("/health")
def health():
    return "ok", 200

# ----------------------------
# Before Request: Enforce Login
# ----------------------------
@app.before_request
def require_login():
    """Redirect to login page if the user isn't logged in and tries to access restricted routes."""
    # Allowed endpoints that do NOT require login
    allowed_routes = {
        "health",
        "login_page", "login", "register", "static",
        "chatbot", "on_submit_query",
        "ai_engine_page", "mobile_device_detected_page", "contact",
        "crop_form", "index", "home_page",
        "buy_now",
    }
    # When gunicorn may call with None endpoint (static files etc.)
    if request.endpoint is None:
        return
    if request.endpoint not in allowed_routes and "user" not in session:
        # Avoid redirect loop if already on login_page
        if request.endpoint != "login_page":
            flash("Please log in to access that page.", "warning")
            return redirect(url_for("login_page"))

# ============================================================
# ============= Lazy loaders to reduce memory =================
# ============================================================
_disease_info = None
_supplement_info = None
_cnn_model = None
_crop_model = None

def get_disease_data():
    """Lazy-load disease/supplement CSVs to reduce startup memory."""
    global _disease_info, _supplement_info
    if _disease_info is None or _supplement_info is None:
        # Import pandas lazily
        import pandas as pd
        _disease_info = pd.read_csv("disease_info.csv", encoding="cp1252")
        _supplement_info = pd.read_csv("supplement_info.csv", encoding="cp1252")
    return _disease_info, _supplement_info

def load_cnn_model():
    """
    Lazy-load CNN model and PyTorch only when needed.
    Falls back gracefully if CNN.py or model file is missing.
    """
    global _cnn_model
    if _cnn_model is not None:
        return _cnn_model

    try:
        # Heavy imports inside
        import torch
        # Import your CNN definition (ensure CNN.py exists in repo root or package path)
        import CNN  # noqa: F401

        model = CNN.CNN(39)  # 39 output classes

        # Map to CPU to avoid CUDA issues on Render
        state = torch.load("plant_disease_model_1_latest.pt", map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _cnn_model = model
        return _cnn_model
    except ModuleNotFoundError:
        # CNN.py missing
        return None
    except FileNotFoundError:
        # We couldn't find the .pt weights
        return None
    except Exception as e:
        # Any other error — keep None and report later
        print("Error loading CNN model:", e)
        return None

def get_crop_model():
    """Lazy-load crop recommendation model."""
    global _crop_model
    if _crop_model is not None:
        return _crop_model
    try:
        with open("crop_recommendation_model.pkl", "rb") as f:
            _crop_model = pickle.load(f)
        return _crop_model
    except FileNotFoundError:
        return None

# ============================================================
# ======================= Routes =============================
# ============================================================

# ----------------------------
# Razorpay
# ----------------------------
@app.route("/buy_now", methods=["POST"])
def buy_now():
    data = request.get_json(force=True)
    payment_id = data.get("payment_id")
    # product_id = data.get('product_id')  # available if you need it

    if not payment_id:
        return jsonify(success=False, error="payment_id required"), 400

    try:
        # Verify payment exists on Razorpay
        razorpay_client.payment.fetch(payment_id)
        # TODO: save order details in DB if needed
        return jsonify(success=True)
    except Exception as e:
        return jsonify(success=False, error=str(e)), 400

# ----------------------------
# Authentication Routes
# ----------------------------
@app.route("/")
def login_page():
    """Landing page (login form)."""
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def login():
    user_type = request.form.get("user_type", "").strip().lower()
    username = request.form.get("username", "").strip().lower()
    password = request.form.get("password")

    if user_type in ["farmer", "buyer", "worker"]:
        collection = (
            mongo.db.farmers if user_type == "farmer"
            else mongo.db.buyers if user_type == "buyer"
            else mongo.db.workers
        )
        user = collection.find_one({"username": username})
        if user and user.get("password") == password:
            session["user"] = username
            session["user_type"] = user_type
            flash("Login successful.", "success")
            if user_type == "farmer":
                return redirect(url_for("base"))
            elif user_type == "buyer":
                return redirect(url_for("buyer_dashboard"))
            else:
                return redirect(url_for("view_work"))
        else:
            flash("Invalid credentials, try again.", "danger")
            return redirect(url_for("login_page"))
    else:
        flash("Invalid user type specified.", "danger")
        return redirect(url_for("login_page"))

@app.route("/register", methods=["GET", "POST"])
def register():
    """Registration route for farmers, buyers, or workers."""
    if request.method == "POST":
        user_type = request.form.get("role", "").strip().lower()
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password")

        if user_type == "farmer":
            mongo.db.farmers.insert_one({"username": username, "password": password})
        elif user_type == "buyer":
            mongo.db.buyers.insert_one({"username": username, "password": password})
        elif user_type == "worker":
            mongo.db.workers.insert_one({"username": username, "password": password})
        else:
            flash("Invalid role specified.", "danger")
            return redirect(url_for("register"))

        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login_page"))

    return render_template("register.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    session.pop("user_type", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("login_page"))

# ----------------------------
# Marketplace Routes
# ----------------------------
@app.route("/farmer_dashboard")
def farmer_dashboard():
    products = mongo.db.products.find({"uploaded_by": session["user"]})
    return render_template("farmer_dashboard.html", products=products)

@app.route("/buyer_dashboard")
def buyer_dashboard():
    products = mongo.db.products.find()
    return render_template("buyer_dashboard.html", products=products)

@app.route("/view_work")
def view_work():
    work_posts = mongo.db.work_posts.find()
    return render_template("view_work.html", work_posts=work_posts)

@app.route("/govr_schemes")
def govr_schemes():
    return render_template("govr_schemes.html")

@app.route("/add_product", methods=["POST"])
def add_product():
    name = request.form.get("name")
    description = request.form.get("description")
    price = request.form.get("price")
    email = request.form.get("email")
    phone = request.form.get("phone")
    bank_account_number = request.form.get("bank_account_number")
    location = request.form.get("location")

    mongo.db.products.insert_one({
        "name": name,
        "description": description,
        "price": price,
        "email": email,
        "phone": phone,
        "bank_account_number": bank_account_number,
        "location": location,
        "uploaded_by": session["user"],
    })
    flash("Product added successfully.", "success")
    return redirect(url_for("farmer_dashboard"))

@app.route("/remove_product", methods=["POST"])
def remove_product():
    product_id = request.form.get("product_id")
    mongo.db.products.delete_one({"_id": ObjectId(product_id)})
    flash("Product removed successfully.", "success")
    return redirect(url_for("farmer_dashboard"))

@app.route("/add_to_cart", methods=["POST"])
def add_to_cart():
    product_id = request.form.get("product_id")
    product = mongo.db.products.find_one({"_id": ObjectId(product_id)})
    if product:
        cart_item = {
            "buyer": session["user"],
            "product_id": product["_id"],
            "name": product.get("name", ""),
            "price": product.get("price", ""),
            "description": product.get("description", ""),
            "email": product.get("email", ""),
            "phone": product.get("phone", ""),
            "bank_account_number": product.get("bank_account_number", ""),
            "location": product.get("location", ""),
        }
        mongo.db.cart.insert_one(cart_item)
        flash("Product added to cart.", "success")
    else:
        flash("Product not found.", "danger")
    return redirect(url_for("buyer_dashboard"))

@app.route("/cart")
def cart():
    cart_items = list(mongo.db.cart.find({"buyer": session["user"]}))
    return render_template("cart.html", cart_items=cart_items)

@app.route("/remove_from_cart", methods=["POST"])
def remove_from_cart():
    cart_item_id = request.form.get("cart_item_id")
    mongo.db.cart.delete_one({"_id": ObjectId(cart_item_id)})
    flash("Item removed from cart.", "info")
    return redirect(url_for("cart"))

# ----------------------------
# Work Posting and Application Routes
# ----------------------------
@app.route("/post_work", methods=["GET", "POST"])
def post_work():
    if request.method == "POST":
        title = request.form.get("title")
        description = request.form.get("description")
        location = request.form.get("location")
        payment = request.form.get("payment")

        mongo.db.work_posts.insert_one({
            "title": title,
            "description": description,
            "location": location,
            "payment": payment,
            "posted_by": session["user"],
        })
        flash("Work posted successfully.", "success")

    return render_template("post_work.html")

@app.route("/remove_work/<work_id>", methods=["POST"])
def remove_work(work_id):
    mongo.db.work_posts.delete_one({"_id": ObjectId(work_id)})
    flash("Work removed successfully.", "success")
    return redirect(url_for("view_work"))

@app.route("/apply_work", methods=["POST"])
def apply_work():
    work_id = request.form.get("work_id")
    name = request.form.get("name")
    email = request.form.get("email")
    phone = request.form.get("phone")
    work_post = mongo.db.work_posts.find_one({"_id": ObjectId(work_id)})
    if work_post:
        application = {
            "worker": session["user"],
            "work_id": work_post["_id"],
            "title": work_post.get("title", ""),
            "description": work_post.get("description", ""),
            "location": work_post.get("location", ""),
            "payment": work_post.get("payment", ""),
            "posted_by": work_post.get("posted_by", ""),
            "name": name,
            "email": email,
            "phone": phone,
        }
        mongo.db.work_applications.insert_one(application)
        farmer = mongo.db.farmers.find_one({"username": work_post.get("posted_by", "")})
        if farmer:
            flash(f"Application submitted successfully. The farmer {farmer['username']} has been notified.", "success")
        else:
            flash("Application submitted successfully.", "success")
    else:
        flash("Work post not found.", "danger")
    return redirect(url_for("view_work"))

# ----------------------------
# Agri-Assistant Pages
# ----------------------------
@app.route("/base")
def base():
    return render_template("base.html")

@app.route("/home")
def home_page():
    return render_template("home.html")

@app.route("/contact")
def contact():
    return render_template("contact-us.html")

@app.route("/index")
def ai_engine_page():
    return render_template("index.html")

@app.route("/mobile-device")
def mobile_device_detected_page():
    return render_template("mobile-device.html")

@app.route("/crop_form")
def crop_form():
    return render_template("crop_form.html")

# ----------------------------
# Disease Prediction
# ----------------------------
def prediction(image_path):
    """
    Run CNN prediction. Loads heavy deps lazily.
    Returns: index (int) or raises RuntimeError with human-friendly message.
    """
    model = load_cnn_model()
    if model is None:
        raise RuntimeError(
            "Model is not available on the server. Ensure CNN.py and plant_disease_model_1_latest.pt are deployed."
        )

    # Heavy import inside
    from PIL import Image
    try:
        # Import only the needed function to keep memory low
        from torchvision.transforms.functional import to_tensor
    except Exception as e:
        raise RuntimeError(f"torchvision is required for prediction: {e}")

    import torch  # local import

    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    input_data = to_tensor(image).unsqueeze(0)  # Add batch dimension
    with torch.inference_mode():
        output = model(input_data)
        index = int(output.detach().cpu().numpy().argmax())
    return index

@app.route("/submit", methods=["GET", "POST"])
def submit():
    if request.method == "POST":
        image = request.files.get("image")
        if not image or not image.filename:
            flash("Please upload an image.", "warning")
            return render_template("submit.html")

        filename = image.filename
        file_path = os.path.join("static", "uploads", filename)
        image.save(file_path)

        try:
            pred = prediction(file_path)
        except RuntimeError as e:
            flash(str(e), "danger")
            return render_template("submit.html")

        disease_info, supplement_info = get_disease_data()

        title = disease_info["disease_name"][pred]
        description = disease_info["description"][pred]
        prevent = disease_info["Possible Steps"][pred]
        image_url = disease_info["image_url"][pred]
        supplement_name = supplement_info["supplement name"][pred]
        supplement_image_url = supplement_info["supplement image"][pred]
        supplement_buy_link = supplement_info["buy link"][pred]

        return render_template(
            "submit.html",
            title=title,
            desc=description,
            prevent=prevent,
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image_url,
            buy_link=supplement_buy_link,
        )
    return render_template("submit.html")

@app.route("/market", methods=["GET", "POST"])
def market():
    disease_info, supplement_info = get_disease_data()
    return render_template(
        "market.html",
        supplement_image=list(supplement_info["supplement image"]),
        supplement_name=list(supplement_info["supplement name"]),
        disease=list(disease_info["disease_name"]),
        buy=list(supplement_info["buy link"]),
    )

# ----------------------------
# Chatbot
# ----------------------------
@app.route("/chatbot")
def chatbot():
    session.pop("history", None)
    session["history"] = []
    welcome_message = "AgriBot: Hello! I am your agricultural assistant. How can I assist you with farming today?"
    session["history"].append({"message": welcome_message, "sender": "bot"})
    return render_template("chatbot.html", history=session["history"])

@app.route("/submit_query", methods=["POST"])
def on_submit_query():
    query = request.form["query"]
    session.setdefault("history", []).append({"message": query, "sender": "user"})

    response = generate_response(query)
    response_message = f"AgriBot Response: {response}"
    session["history"].append({"message": response_message, "sender": "bot"})

    return jsonify({"query": query, "response": response_message})

def generate_response(query):
    qa_prompt = (
        "You are an intelligent agriculture assistant designed to provide accurate and actionable advice to "
        "farmers and agriculture enthusiasts. Your primary role is to guide users on various aspects of farming, "
        "including crop selection, climate conditions, soil health, pest control, irrigation practices, and sustainable "
        "agricultural techniques. Additionally, you offer insights on weather forecasts, market trends, and best practices "
        "to help users optimize their farming operations. Your responses should be clear and simple, tailored to users with "
        "different levels of expertise, and provide localized advice when possible by considering the user’s region and climate. "
        "Always ensure that your answers are based on the latest agricultural knowledge and research, and offer follow-up "
        "recommendations to support users in making well-informed decisions."
    )

    chat_history = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in session.get("history", [])])
    input_text = f"{qa_prompt}\n{chat_history}\nUser question:\n{query}"

    agriculture_keywords = [
        "crop", "farming", "soil", "irrigation", "pest", "climate", "weather", "agriculture", "plant", "harvest",
        "fertilizer", "yield", "sustainable", "organic", "farm", "livestock", "disease", "pesticide", "herbicide",
        "compost", "greenhouse", "horticulture", "agronomy", "agroforestry", "aquaculture", "beekeeping", "dairy",
        "forestry", "hydroponics", "permaculture", "silviculture", "viticulture", "agricultural"
    ]

    if any(keyword in query.lower() for keyword in agriculture_keywords):
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        result = llm.invoke(input_text)
        return result.content
    else:
        return "I'm sorry, I can only assist with agricultural-related questions. Please ask me something related to farming or agriculture."

# ----------------------------
# Crop Recommendation
# ----------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    if request.method == "POST":
        try:
            N = float(request.form["N"])
            P = float(request.form["P"])
            K = float(request.form["K"])
            temperature = float(request.form["temperature"])
            humidity = float(request.form["humidity"])
            ph = float(request.form["ph"])
            rainfall = float(request.form["rainfall"])
        except (KeyError, ValueError):
            flash("Please provide all numeric inputs correctly.", "danger")
            return render_template("result.html", crop=None)

        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        model = get_crop_model()
        if model is None:
            flash("Crop model not available on the server.", "danger")
            return render_template("result.html", crop=None)

        predicted_crop = model.predict(input_features)[0]
        return render_template(
            "result.html",
            crop=predicted_crop,
            image_url=url_for("static", filename=f"images/{str(predicted_crop).lower()}.jpg"),
            instructions="Plant the seeds in well-drained soil with proper spacing and irrigation. Use organic fertilizer.",
            yield_info="Expected yield is approximately 2-3 tons per acre.",
        )

# ----------------------------
# Run the App (Render binds PORT)
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # host=0.0.0.0 is required for Render
    app.run(host="0.0.0.0", port=port)

