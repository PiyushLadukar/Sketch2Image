# 🎨 SketchVerse AI

> **Transform hand-drawn sketches into realistic AI-generated images using Computer Vision, Retrieval-Augmented Generation (RAG), and Diffusion Models.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Next.js](https://img.shields.io/badge/Next.js-Frontend-black)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Under%20Development-orange)

---

## 📖 About the Project

SketchVerse AI is an AI-powered web application that converts user-drawn sketches into realistic images.

Users can draw directly on an interactive paint canvas, describe their sketch using a prompt, and generate AI-powered images locally without relying on paid third-party APIs.

The application combines Computer Vision, Retrieval-Augmented Generation (RAG), and Diffusion Models to improve image quality while preserving the original sketch structure.

---

## ✨ Features

* 🎨 Interactive Paint Canvas
* ✏️ Brush, Pencil, Eraser & Shape Tools
* 💬 Prompt Input for Detailed Image Description
* 🤖 AI-Powered Sketch-to-Image Generation
* 📚 RAG-Based Prompt Enhancement
* 🧠 Local AI Model Inference
* 🖼️ Image Preview & Download
* 📂 Project History
* 🌙 Dark Mode (Planned)

---

## 🛠️ Tech Stack

### Frontend

* Next.js
* React
* TypeScript
* Tailwind CSS
* Fabric.js

### Backend

* FastAPI
* Python

### AI / Machine Learning

* OpenCV
* PyTorch
* Hugging Face Transformers
* Diffusers
* ControlNet
* Stable Diffusion XL
* Sentence Transformers

### Database

* PostgreSQL
* ChromaDB

---

## 🏗️ Architecture

```text
User
   │
   ▼
Paint Canvas + Prompt
   │
   ▼
FastAPI Backend
   │
   ├── OpenCV
   ├── RAG
   ├── Prompt Builder
   ├── ControlNet
   └── Stable Diffusion XL
   │
   ▼
Generated Image
```

---

## 📂 Project Structure

```text
SketchVerseAI/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── canvas/
│   └── styles/
│
├── backend/
│   ├── api/
│   ├── ai/
│   ├── rag/
│   ├── preprocessing/
│   ├── generation/
│   └── database/
│
├── docs/
├── generated_images/
├── requirements.txt
├── package.json
└── README.md
```

---

## ⚙️ How It Works

1. User draws a sketch on the canvas.
2. User enters a descriptive prompt.
3. The sketch is preprocessed using OpenCV.
4. The prompt is enhanced using RAG.
5. ControlNet preserves the sketch structure.
6. Stable Diffusion XL generates a realistic image.
7. The final image is displayed and can be downloaded.

---

## 🚀 Getting Started

### Clone the Repository

```bash
git clone https://github.com/your-username/SketchVerseAI.git
cd SketchVerseAI
```

### Backend Setup

```bash
cd backend
python -m venv venv
```

Activate the virtual environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the backend:

```bash
uvicorn main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## 📌 Roadmap

* [ ] Interactive Paint Application
* [ ] AI Prompt Assistant
* [ ] RAG Integration
* [ ] Stable Diffusion XL Integration
* [ ] ControlNet Support
* [ ] Image History
* [ ] User Authentication
* [ ] Image Upscaling
* [ ] Multi-Style Generation
* [ ] Cloud Deployment



