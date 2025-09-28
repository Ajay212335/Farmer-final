from flask import Flask, render_template, redirect, url_for, request, session, flash, jsonify
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
from PIL import Image
import torchvision.transforms.functional as TF
import CNN  # Ensure you have the CNN module properly set up in your project
import numpy as np
import torch
import pandas as pd
from flask_session import Session
from dotenv import load_dotenv
import pickle
import os
import razorpay
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)

# ----------------------------
# Razorpay Client Initialization
# ----------------------------
razorpay_client = razorpay.Client(auth=("YOUR_RAZORPAY_KEY_ID", "YOUR_RAZORPAY_SECRET"))

@app.route('/buy_now', methods=['POST'])
def buy_now():
    data = request.get_json()
    payment_id = data['payment_id']
    product_id = data.get('product_id')  # Just in case you need product_id

    # Verify the payment (optional)
    try:
        razorpay_client.payment.fetch(payment_id)
        # Process the order here (e.g., save order details to the database)
        return jsonify(success=True)
    except:
        return jsonify(success=False)

# ----------------------------
# Environment Setup
# ----------------------------
load_dotenv()  # Load .env file if present

# Optionally set environment variables here
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app.secret_key = 'your_secret_key_here'

# ----------------------------
# Session Configuration
# ----------------------------
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# ----------------------------
# MongoDB Setup
# ----------------------------
app.config["MONGO_URI"] = "mongodb://localhost:27017/loginapp"
mongo = PyMongo(app)

# ----------------------------
# Before Request: Enforce Login
# ----------------------------
@app.before_request
def require_login():
    """Redirect to login page if the user isn't logged in and tries to access restricted routes."""
    # Allowed endpoints that do NOT require login
    allowed_routes = {
        'login_page', 'login', 'register', 'static'
    }
    # If endpoint not in allowed_routes and user not in session => redirect
    if request.endpoint not in allowed_routes and 'user' not in session:
        flash("Please log in to access that page.", "warning")
        return redirect(url_for('login_page'))

# ----------------------------
# Load Data and Models for Agri-Assistant
# ----------------------------
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# CNN Model for Disease Prediction
cnn_model = CNN.CNN(39)  # 39 output classes
cnn_model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
cnn_model.eval()

# Crop Recommendation Model (pickle)
with open('crop_recommendation_model.pkl', 'rb') as file:
    crop_model = pickle.load(file)

# ----------------------------
# Authentication Routes
# ----------------------------
@app.route('/')
def login_page():
    """Landing page (login form)."""
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    user_type = request.form.get('user_type', '').strip().lower()
    username = request.form.get('username', '').strip().lower()
    password = request.form.get('password')

    if user_type in ['farmer', 'buyer', 'worker']:
        # Marketplace login
        collection = mongo.db.farmers if user_type == 'farmer' else mongo.db.buyers if user_type == 'buyer' else mongo.db.workers
        user = collection.find_one({'username': username})
        if user and user['password'] == password:
            session['user'] = username
            session['user_type'] = user_type
            flash("Login successful.", "success")
            # Redirect to the appropriate dashboard
            if user_type == 'farmer':
                return redirect(url_for('base'))
            elif user_type == 'buyer':
                return redirect(url_for('buyer_dashboard'))
            else:
                return redirect(url_for('view_work'))
        else:
            flash("Invalid credentials, try again.", "danger")
            return redirect(url_for('login_page'))
    else:
        flash("Invalid user type specified.", "danger")
        return redirect(url_for('login_page'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration route for farmers, buyers, or workers."""
    if request.method == 'POST':
        user_type = request.form.get('role', '').strip().lower()
        username = request.form.get('username').strip().lower()
        password = request.form.get('password')

        if user_type == 'farmer':
            mongo.db.farmers.insert_one({'username': username, 'password': password})
        elif user_type == 'buyer':
            mongo.db.buyers.insert_one({'username': username, 'password': password})
        elif user_type == 'worker':
            mongo.db.workers.insert_one({'username': username, 'password': password})
        else:
            flash("Invalid role specified.", "danger")
            return redirect(url_for('register'))

        flash("Registration successful. Please log in.", "success")
        return redirect(url_for('login_page'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    session.pop('user_type', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login_page'))

# ----------------------------
# Marketplace Routes
# ----------------------------
@app.route('/farmer_dashboard')
def farmer_dashboard():
    products = mongo.db.products.find({'uploaded_by': session['user']})
    return render_template('farmer_dashboard.html', products=products)

@app.route('/buyer_dashboard')
def buyer_dashboard():
    products = mongo.db.products.find()
    return render_template('buyer_dashboard.html', products=products)

@app.route('/view_work')
def view_work():
    work_posts = mongo.db.work_posts.find()
    return render_template('view_work.html', work_posts=work_posts)

@app.route('/govr_schemes')
def govr_schemes():
    return render_template('govr_schemes.html')

@app.route('/add_product', methods=['POST'])
def add_product():
    name = request.form.get('name')
    description = request.form.get('description')
    price = request.form.get('price')
    email = request.form.get('email')
    phone = request.form.get('phone')
    bank_account_number = request.form.get('bank_account_number')
    location = request.form.get('location')

    mongo.db.products.insert_one({
        'name': name,
        'description': description,
        'price': price,
        'email': email,
        'phone': phone,
        'bank_account_number': bank_account_number,
        'location': location,
        'uploaded_by': session['user']
    })
    flash("Product added successfully.", "success")
    return redirect(url_for('farmer_dashboard'))

@app.route('/remove_product', methods=['POST'])
def remove_product():
    product_id = request.form.get('product_id')
    mongo.db.products.delete_one({'_id': ObjectId(product_id)})
    flash("Product removed successfully.", "success")
    return redirect(url_for('farmer_dashboard'))

@app.route('/add_to_cart', methods=['POST'])
def add_to_cart():
    product_id = request.form.get('product_id')
    product = mongo.db.products.find_one({'_id': ObjectId(product_id)})
    if product:
        cart_item = {
            'buyer': session['user'],
            'product_id': product['_id'],
            'name': product.get('name', ''),
            'price': product.get('price', ''),
            'description': product.get('description', ''),
            'email': product.get('email', ''),
            'phone': product.get('phone', ''),
            'bank_account_number': product.get('bank_account_number', ''),
            'location': product.get('location', ''),
        }
        mongo.db.cart.insert_one(cart_item)
        flash("Product added to cart.", "success")
    else:
        flash("Product not found.", "danger")
    return redirect(url_for('buyer_dashboard'))

@app.route('/cart')
def cart():
    cart_items = list(mongo.db.cart.find({'buyer': session['user']}))
    return render_template('cart.html', cart_items=cart_items)

@app.route('/remove_from_cart', methods=['POST'])
def remove_from_cart():
    cart_item_id = request.form.get('cart_item_id')
    mongo.db.cart.delete_one({'_id': ObjectId(cart_item_id)})
    flash("Item removed from cart.", "info")
    return redirect(url_for('cart'))

# ----------------------------
# Work Posting and Application Routes
# ----------------------------
@app.route('/post_work', methods=['GET', 'POST'])
def post_work():
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        location = request.form.get('location')
        payment = request.form.get('payment')

        mongo.db.work_posts.insert_one({
            'title': title,
            'description': description,
            'location': location,
            'payment': payment,
            'posted_by': session['user']
        })
        flash("Work posted successfully.", "success")
        
    
    return render_template('post_work.html')

@app.route('/remove_work/<work_id>', methods=['POST'])
def remove_work(work_id):
    mongo.db.work_posts.delete_one({'_id': ObjectId(work_id)})
    flash("Work removed successfully.", "success")
    return redirect(url_for('view_work'))

@app.route('/apply_work', methods=['POST'])
def apply_work():
    work_id = request.form.get('work_id')
    name = request.form.get('name')
    email = request.form.get('email')
    phone = request.form.get('phone')
    work_post = mongo.db.work_posts.find_one({'_id': ObjectId(work_id)})
    if work_post:
        application = {
            'worker': session['user'],
            'work_id': work_post['_id'],
            'title': work_post.get('title', ''),
            'description': work_post.get('description', ''),
            'location': work_post.get('location', ''),
            'payment': work_post.get('payment', ''),
            'posted_by': work_post.get('posted_by', ''),
            'name': name,
            'email': email,
            'phone': phone
        }
        mongo.db.work_applications.insert_one(application)
        # Notify the farmer who posted the work
        farmer = mongo.db.farmers.find_one({'username': work_post.get('posted_by', '')})
        if farmer:
            # Here you can implement the notification logic, e.g., send an email or message
            flash(f"Application submitted successfully. The farmer {farmer['username']} has been notified.", "success")
        else:
            flash("Application submitted successfully.", "success")
    else:
        flash("Work post not found.", "danger")
    return redirect(url_for('view_work'))

# ----------------------------
# Agri-Assistant Routes
# ----------------------------
@app.route('/base')
def base():
    return render_template('base.html')

@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/crop_form')
def crop_form():
    return render_template('crop_form.html')

# ----------------------------
# Disease Prediction
# ----------------------------
def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).unsqueeze(0)  # Add batch dimension
    output = cnn_model(input_data)
    output = output.detach().numpy()
    index = np.argmax(output)
    return index

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        
        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent, 
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image_url,
            buy_link=supplement_buy_link
        )
    return render_template('submit.html')

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']), 
        disease=list(disease_info['disease_name']), 
        buy=list(supplement_info['buy link'])
    )

# ----------------------------
# Chatbot
# ----------------------------
@app.route('/chatbot')
def chatbot():
    # Initialize chat history for a fresh start
    session.pop('history', None)
    session['history'] = []
    welcome_message = "AgriBot: Hello! I am your agricultural assistant. How can I assist you with farming today?"
    session['history'].append({'message': welcome_message, 'sender': 'bot'})
    return render_template('chatbot.html', history=session['history'])

@app.route('/submit_query', methods=['POST'])
def on_submit_query():
    query = request.form['query']
    session.setdefault('history', []).append({'message': query, 'sender': 'user'})
    
    response = generate_response(query)
    response_message = f"AgriBot Response: {response}"
    session['history'].append({'message': response_message, 'sender': 'bot'})
    
    return jsonify({'query': query, 'response': response_message})

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
    
    # Include chat history in the prompt
    chat_history = "\n".join([f"{msg['sender']}: {msg['message']}" for msg in session['history']])
    input_text = f"{qa_prompt}\n{chat_history}\nUser question:\n{query}"
    
    # Check if the query is related to agriculture
    agriculture_keywords = [
        "crop", "farming", "soil", "irrigation", "pest", "climate", "weather", "agriculture", "plant", "harvest",
        "fertilizer", "yield", "sustainable", "organic", "farm", "livestock", "disease", "pesticide", "herbicide",
        "compost", "greenhouse", "horticulture", "agronomy", "agroforestry", "aquaculture", "beekeeping", "dairy",
        "forestry", "hydroponics", "livestock", "permaculture", "silviculture", "viticulture", "agricultural"
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
@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        predicted_crop = crop_model.predict(input_features)[0]
        return render_template(
            'result.html',
            crop=predicted_crop,
            image_url=url_for('static', filename=f'images/{predicted_crop.lower()}.jpg'),
            instructions="Plant the seeds in well-drained soil with proper spacing and irrigation. Use organic fertilizer.",
            yield_info="Expected yield is approximately 2-3 tons per acre."
        )

# ----------------------------
# Run the App
# ----------------------------
if __name__ == '__main__':
    app.run(debug=True)