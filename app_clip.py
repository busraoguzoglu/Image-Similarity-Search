import os
import numpy as np
import torch
import clip
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from sklearn.metrics.pairwise import cosine_similarity

# Directory containing images
image_dir = 'test_images_design_subset'
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

app = Flask(__name__)

# Load and preprocess images
def load_images(image_dir):
    image_list = []
    file_names = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path).convert("RGB")
            image_list.append(img)
            file_names.append(filename)
    return image_list, file_names

images, file_names = load_images(image_dir)
embeddings = None

def get_image_embeddings(model, preprocess, images):
    image_tensors = [preprocess(img).unsqueeze(0).to(device) for img in images]
    image_features = []
    with torch.no_grad():
        for image_tensor in image_tensors:
            image_features.append(model.encode_image(image_tensor))
    embeddings = torch.cat(image_features).numpy()
    return embeddings

embeddings = get_image_embeddings(model, preprocess, images)

"""@app.route('/similar_images_text', methods=['POST'])
def similar_images_text():
    data = request.json
    query_text = data['query']
    text_inputs = clip.tokenize([query_text]).to(device)
    
    with torch.no_grad():
        query_embedding = model.encode_text(text_inputs).cpu().numpy()
        
    
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    top_k_indices = similarities.argsort()[-6:][::-1]
    
    similar_images = [file_names[idx] for idx in top_k_indices]
    
    return jsonify(similar_images)

"""

@app.route('/similar_images_text', methods=['POST'])
def similar_images_text():
    data = request.json
    query_text = data['query']
    negative_text = data.get('negative_query', '')  # Get the negative prompt if provided
    
    text_inputs = clip.tokenize([query_text]).to(device)
    negative_text_inputs = clip.tokenize([negative_text]).to(device)
    
    with torch.no_grad():
        query_embedding = model.encode_text(text_inputs).cpu().numpy()
        negative_embedding = model.encode_text(negative_text_inputs).cpu().numpy()
    
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    negative_similarities = cosine_similarity(negative_embedding, embeddings).flatten()
    
    combined_similarities = similarities - negative_similarities  # Adjust the similarities with negative prompt
    top_k_indices = combined_similarities.argsort()[-6:][::-1]
    
    similar_images = [file_names[idx] for idx in top_k_indices]
    
    return jsonify(similar_images)


@app.route('/similar_images_upload', methods=['POST'])
def similar_images_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    img = Image.open(file.stream).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        query_embedding = model.encode_image(img_tensor).cpu().numpy()
    
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    top_k_indices = similarities.argsort()[-6:][::-1]
    
    similar_images = [file_names[idx] for idx in top_k_indices]
    
    print(f"Similar images: {similar_images}")  # Log the similar images paths
    
    return jsonify(similar_images)

@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(image_dir, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
