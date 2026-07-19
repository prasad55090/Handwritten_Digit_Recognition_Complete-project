from app import app, get_digit_model, get_text_model

if __name__ == '__main__':
    # Initialize the models on startup
    get_digit_model()
    get_text_model()
    # Start the Flask development server on port 5001
    app.run(host='0.0.0.0', port=5001, debug=True)
