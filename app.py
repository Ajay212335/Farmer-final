from flask import Flask, request, jsonify

app = Flask(__name__)

# Test route
@app.route("/")
def home():
    return jsonify({"message": "Backend is running!"})

# Example route to receive data
@app.route("/data", methods=["POST"])
def receive_data():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    return jsonify({"message": "Data received successfully", "data": data})

# Example route to send a response
@app.route("/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello from backend!"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
