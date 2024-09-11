import os
import numpy as np
import torch
import clip
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from sklearn.metrics.pairwise import cosine_similarity
from qdrant_client.models import VectorParams
import logging

# Directory containing images
image_dir = 'test_images_design_subset'
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

app = Flask(__name__)
app.config['DEBUG'] = True

###################################################################################
###########################Qdrant Initialization###################################

# Initialize Qdrant client
qdrant_client = QdrantClient(url="http://localhost:6333")

# Creating collection in Qdrant
def get_all_collections():
    try:
        collections_list = []
        collections = qdrant_client.get_collections()
        for collection in collections:
            for c in list(collection[1]):
                collections_list.append(c.name)
        return collections_list
    except Exception as e:
        print(f"Error fetching collections from Qdrant: {e}")

collections = get_all_collections()
    
if "images" not in collections:
    qdrant_client.create_collection(
        collection_name="images",
        vectors_config=VectorParams(size=512, distance=rest.Distance.COSINE)
    )

###################################################################################

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

def get_image_embeddings(model, preprocess, images):
    image_tensors = [preprocess(img).unsqueeze(0).to(device) for img in images]
    image_features = []
    with torch.no_grad():
        for image_tensor in image_tensors:
            image_features.append(model.encode_image(image_tensor))
    embeddings = torch.cat(image_features).numpy()
    return embeddings

embeddings = get_image_embeddings(model, preprocess, images)

# Store embeddings in Qdrant
# Check if embeddings for images already exist
def check_embeddings_exist():
    try:
        response = qdrant_client.count(collection_name="images")
        embedding_count = response.count
        image_count = len(images)
        return embedding_count >= image_count
    except Exception as e:
        print(f"Error checking embeddings existence: {e}")
        return False
    
def store_embeddings_in_qdrant(embeddings, file_names):
    points = [
        rest.PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={"file_name": file_names[i]}
        ) for i in range(len(file_names))
    ]
    qdrant_client.upsert(
        collection_name="images",
        points=points
    )

# Check if embeddings already exist and store if not
if not check_embeddings_exist():
    print("Embeddings do not exist in Qdrant, creating and storing embeddings...")
    embeddings = get_image_embeddings(model, preprocess, images)
    store_embeddings_in_qdrant(embeddings, file_names)
else:
    print("Embeddings already exist in Qdrant, skipping creation...")


###################################################################################
##################################Functions########################################

@app.route('/similar_images_text', methods=['POST'])
def similar_images_text():
    try:
        data = request.json
        query_text = data['query']
        negative_text = data.get('negative_query', '')  # Get the negative prompt if provided
        
        text_inputs = clip.tokenize([query_text]).to(device)
        negative_text_inputs = clip.tokenize([negative_text]).to(device)
        
        with torch.no_grad():
            query_embedding = model.encode_text(text_inputs).cpu().numpy()
            negative_embedding = model.encode_text(negative_text_inputs).cpu().numpy()
        
        # Combine query embedding with negative embedding
        combined_embedding = query_embedding - negative_embedding

        # Flatten the combined_embedding to a 1D list
        combined_embedding_flat = combined_embedding.flatten().tolist()

        # Search Qdrant for similar embeddings
        search_result = qdrant_client.search(
            collection_name="images",
            query_vector=combined_embedding_flat,
            limit=6
        )

        similar_images = [hit.payload["file_name"] for hit in search_result]
        
        return jsonify(similar_images)
    except Exception as e:
        logging.error("Error in similar_images_text: %s", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/similar_images_upload', methods=['POST'])
def similar_images_upload():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        img = Image.open(file.stream).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            query_embedding = model.encode_image(img_tensor).cpu().numpy()
        
        # Flatten the query_embedding to a 1D list
        query_embedding_flat = query_embedding.flatten().tolist()

        # Search Qdrant for similar embeddings
        search_result = qdrant_client.search(
            collection_name="images",
            query_vector=query_embedding_flat,
            limit=6
        )

        similar_images = [hit.payload["file_name"] for hit in search_result]
        
        print(f"Similar images: {similar_images}")  # Log the similar images paths
        
        return jsonify(similar_images)
    except Exception as e:
        logging.error("Error in similar_images_upload: %s", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/images/<filename>')
def get_image(filename):
    return send_from_directory(image_dir, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)