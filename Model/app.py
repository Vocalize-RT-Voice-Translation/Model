from flask import Flask, request, jsonify
from transformers import MarianMTModel, MarianTokenizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Load models for Hindi to English and English to Hindi translation
model_hi_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-hi-en")
tokenizer_hi_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-hi-en")

model_en_hi = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
tokenizer_en_hi = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")


@app.route('/', methods=['GET'])
def baseRoute():
    return jsonify({'status': 'ok'}), 200   

@app.route('/translate', methods=['POST'])
def translate():
    # Get the JSON payload from the request
    data = request.get_json()
    source = data.get('source')
    target = data.get('target')
    message = data.get('message')

    # Validate inputs
    if not source or not target or not message:
        return jsonify({'error': 'Missing parameters'}), 400

    if source not in ['hi', 'en'] or target not in ['hi', 'en']:
        return jsonify({'error': 'Invalid source or target language'}), 400

    # Choose the appropriate model based on the source and target languages
    if source == 'hi' and target == 'en':
        model = model_hi_en
        tokenizer = tokenizer_hi_en
    elif source == 'en' and target == 'hi':
        model = model_en_hi
        tokenizer = tokenizer_en_hi
    else:
        return jsonify({'error': 'Invalid language pair'}), 400

    # Translate the message
    translated = model.generate(**tokenizer.prepare_seq2seq_batch(message, return_tensors="pt"))
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)

    return jsonify({'translated_message': translation})


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
