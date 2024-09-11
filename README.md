Demo for image similarity project.

## Features

**1- Searching an image via text:** Add positive and negative prompts to search for images. Avoid typing positive prompts such as 'not blue', instead, use the negative prompt.

**2- Searching an image via image:** Upload an image, and you will get similar images in the vector store.

## Notes
- Qdrant is used as vector storage. Run Qdrant locally with Docker image to run the demo.
- CLIP is used as the model. It can embed both text and image in semantic vector space, therefore we can search the similarities.
