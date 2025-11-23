import io
import uuid
from typing import List, Dict, Tuple

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import face_recognition
from sklearn.cluster import DBSCAN

import networkx as nx
from pyvis.network import Network
from streamlit.components.v1 import html

# ---- Utility functions ----

def load_image(file) -> np.ndarray:
    image = Image.open(file).convert("RGB")
    return np.array(image)

def detect_and_embed_faces(image: np.ndarray):
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    faces = []
    for loc, enc in zip(face_locations, face_encodings):
        faces.append({"embedding": enc, "location": loc})
    return faces

def cluster_faces(embeddings: np.ndarray, eps=0.45, min_samples=2):
    if len(embeddings) == 0:
        return np.array([])
    clustering = DBSCAN(metric="euclidean", eps=eps, min_samples=min_samples)
    return clustering.fit_predict(embeddings)

def draw_faces_with_labels(image: np.ndarray, faces, labels):
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for face, label in zip(faces, labels):
        top, right, bottom, left = face["location"]
        color = (0, 255, 0) if label != -1 else (255, 0, 0)
        draw.rectangle(((left, top), (right, bottom)), outline=color, width=3)
        text = f"cluster {label}" if label != -1 else "noise"
        draw.text((left, top - 15), text, fill=color, font=font)

    return pil_img

def build_pyvis_graph(embeddings, labels, threshold=0.6):
    net = Network(height="600px", width="100%", notebook=False, directed=False)
    net.barnes_hut()

    # Assign colors per cluster
    unique_clusters = sorted(set(labels))
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    cluster_color = {c: palette[i % len(palette)] for i, c in enumerate(unique_clusters)}

    # Add nodes
    for i, label in enumerate(labels):
        color = cluster_color[label]
        title = f"Face {i}<br>Cluster {label}"
        net.add_node(
            n_id=i,
            label=str(i),
            color=color,
            title=title,
            size=18
        )

    # Add edges for similar embeddings
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            if dist < threshold:
                net.add_edge(i, j, title=f"dist={dist:.2f}")

    return net


# ---- Streamlit App ----

def main():
    st.title("Face Clustering + Interactive Graph (PyVis)")

    uploaded_files = st.file_uploader(
        "Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    eps = st.slider("DBSCAN eps", 0.2, 1.0, 0.45)
    min_samples = st.slider("DBSCAN min_samples", 1, 5, 2)
    graph_threshold = st.slider("Graph edge threshold", 0.3, 1.5, 0.6)

    if not uploaded_files:
        st.info("Upload at least one image")
        return

    all_embeddings = []
    faces_per_image = []

    for file in uploaded_files:
        st.subheader(f"File: {file.name}")
        img = load_image(file)
        faces = detect_and_embed_faces(img)

        if len(faces) == 0:
            st.warning("No face detected.")
            continue

        embeddings = np.array([f["embedding"] for f in faces])
        faces_per_image.append({
            "name": file.name,
            "image": img,
            "faces": faces,
            "embeddings": embeddings
        })
        all_embeddings.extend(embeddings)

    all_embeddings = np.array(all_embeddings)
    labels = cluster_faces(all_embeddings, eps=eps, min_samples=min_samples)

    # Assign labels back to each image
    idx = 0
    for data in faces_per_image:
        n = len(data["faces"])
        img_labels = labels[idx:idx+n]
        idx += n

        annotated = draw_faces_with_labels(data["image"], data["faces"], img_labels)
        st.image(annotated, caption=f"Clusters in {data['name']}", use_container_width=True)

    # ---- PyVis Graph ----
    st.subheader("Interactive Face Similarity Graph (PyVis)")

    if len(all_embeddings) > 0:
        net = build_pyvis_graph(all_embeddings, labels, threshold=graph_threshold)

        html_code = net.generate_html()
        html(html_code, height=600)
    else:
        st.warning("No embeddings available.")


if __name__ == "__main__":
    main()
