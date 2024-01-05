from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

# Global variable to store the selected model (initialize with a default model)
selected_model = "text-davinci-002"


@app.route('/')
@app.route('/home')
def index():
    return render_template("index.html")


@app.route('/chat')
def chat():
    return render_template("chat.html")

@app.route('/signup')
def signup():
    return render_template("signup.html")

@app.route('/api/generate_response', methods=['POST'])
def generate_response():
    user_message = request.json['message']

    # Call your API or perform any desired processing
    response_message = f"Received: {user_message}. This is a generated response."

    return jsonify({'message': response_message})


@app.route('/api/set_model', methods=['POST'])
def set_model():
    global selected_model
    selected_model = request.json['model']
    return jsonify({'message': f'Selected model set to {selected_model}'})


@app.route('/login')
def login():
    return render_template("login.html")

if __name__ == '__main__':
    app.run(debug=True)

