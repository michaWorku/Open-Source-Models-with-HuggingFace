# Open Source Models with Hugging Face

## Course Overview

The [**Open Source Models with Hugging Face**](https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/) course provides a hands-on introduction to working with open-source AI models using the **Hugging Face ecosystem**. Participants will learn how to find, deploy, and fine-tune models for **NLP, audio, image, and multimodal tasks** using the `transformers` library. Additionally, the course explores **Gradio and Hugging Face Spaces** to create and share AI-powered applications easily.

### Key Topics:
- **Model Selection**: Finding open-source models based on tasks, rankings, and memory requirements.
- **Natural Language Processing (NLP)**: Building chatbots, language translation, text summarization, and sentence similarity analysis.
- **Audio AI**: Implementing **Automatic Speech Recognition (ASR)**, **Text-to-Speech (TTS)**, and **Zero-shot audio classification**.
- **Computer Vision**: Performing **object detection, image segmentation, visual Q&A, and image captioning**.
- **Multimodal AI**: Combining models for advanced tasks like **image-based storytelling**.
- **Deployment**: Packaging and deploying AI apps with **Gradio and Hugging Face Spaces** for easy interaction.

By the end of this course, you will have foundational skills in discovering, implementing, and deploying AI-powered applications. 🚀


## Course Content

### 1. [**NLP**]()
- Introduction to NLP pipelines in Hugging Face.
- Building chatbots using **BlenderBot-400M-Distill**.
- Maintaining conversational context manually.
- Comparing open-source LLMs.

### 2. [**Translation and Summarization**]()
- Translation using **No Language Left Behind (NLLB-200-Distilled-600M)**.
- Summarization with **BART-Large-CNN**.

### 3. [**Sentence Embeddings & Similarity**]()
- Using **all-MiniLM-L6-v2** for sentence embeddings.
- Computing **cosine similarity** between sentences.

### 4. [**Zero-Shot Audio Classification**]()
- Utilizing the **CLAP-HTSAT model** for zero-shot classification.
- Handling audio data preprocessing and correct sampling rates.

### 5. [**Automatic Speech Recognition (ASR)**]()
- Implementing **Distil-Whisper** for speech recognition.
- Deploying ASR with **Gradio**.

### 6. [**Text-to-Speech (TTS)**]()
- Converting text to natural-sounding speech using **VITS (kakao-enterprise/vits-ljs)**.
- Experimenting with phoneme-based synthesis.

### 7. [**Object Detection**]()
- Using **DETR (facebook/detr-resnet-50)** for object detection.
- Deploying detection models with **Gradio**.

### 8. [**Image Segmentation & Depth Estimation**]()
- Utilizing **Meta's SAM (Segment Anything Model)** for segmentation.
- Performing **depth estimation with DPT**.

### 9. [**Image Retrieval**]()
- Matching images with text descriptions using **BLIP (Bootstrapped Language-Image Pretraining)**.

### 10. [**Image Captioning**]()
- Generating image captions using the **BLIP model**.

### 11. [**Visual Question & Answering**]()
- Demonstrating Visual Question Answering (VQA) using the **BLIP model**.

### 12. [**Zero-Shot Image Classification**]()
- Demonstrating Zero-Shot image classification using the **CLIP model**.

### 13. [**Deployment**]()
- deploying ML models on **Hugging Face Spaces** using **Gradio**.


## Notebooks
The course includes interactive Jupyter Notebooks covering each topic. The notebooks contain:
- Code implementations.
- Hands-on exercises.
- Step-by-step guidance on using **Hugging Face** models.


## Getting Started

### 1. **Installation**
To run the notebooks locally, install the necessary libraries:
```bash
pip install transformers datasets gradio soundfile librosa timm inflect phonemizer sentence-transformers
```


### 2. **Running the Notebooks**
- Open Jupyter Notebook or Google Colab.
- Clone the course repository or upload the notebooks.
- Run each cell and follow the guided exercises.

### 3. **Deploying Models**
- Utilize **Gradio** to create interactive AI applications.
- Deploy models on **Hugging Face Spaces**.



## Resources and References
- [**Open Source Models with Hugging Face**](https://www.deeplearning.ai/short-courses/open-source-models-hugging-face/)
- **Hugging Face Hub**: [https://huggingface.co/models](https://huggingface.co/models)
- **Hugging Face Transformers Documentation**: [https://huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
- **Gradio Documentation**: [https://www.gradio.app](https://www.gradio.app)
- **Open LLM Leaderboard**: [https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- **LMSYS Chatbot Arena**: [https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)

Happy Learning! 🚀

