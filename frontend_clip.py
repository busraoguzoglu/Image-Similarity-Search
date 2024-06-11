import os
import gradio as gr
import requests

# Function to find similar images based on text query with negative prompt
def find_similar_images_text(query, negative_query):
    print(f"Query: {query}, Negative Query: {negative_query}")
    response = requests.post("http://localhost:5000/similar_images_text", json={"query": query, "negative_query": negative_query})
    if response.status_code == 200:
        similar_images = response.json()
        print(f"Text Query Similar Images: {similar_images}")  # Log the response
        return [f"http://localhost:5000/images/{img}" for img in similar_images]
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return []

# Function to find similar images based on an uploaded image
def find_similar_images_upload(image_path):
    print ("PATH: ", image_path)
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post("http://localhost:5000/similar_images_upload", files=files)
    if response.status_code == 200:
        similar_images = response.json()
        print(f"Image Upload Similar Images: {similar_images}")  # Log the response
        return [f"http://localhost:5000/images/{img}" for img in similar_images]
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return []

# Gradio Interface
iface_text = gr.Interface(
    fn=find_similar_images_text,
    inputs=[gr.Textbox(label="Enter query text"), gr.Textbox(label="Enter negative query text")],
    outputs=gr.Gallery(label="Similar Images")
)

iface_upload = gr.Interface(
    fn=find_similar_images_upload,
    # inputs=gr.Image(type="filepath", label="Upload an image"),
    inputs=gr.Image(type="filepath"),
    outputs=gr.Gallery(label="Similar Images")
)

app = gr.TabbedInterface([iface_text, iface_upload], ["Text Search", "Image Upload"])

if __name__ == "__main__":
    app.launch()
