import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"

import base64
import io
import tempfile
import streamlit as st
import numpy as np
from PIL import Image

from deepface import DeepFace
from streamlit_agraph import agraph, Node, Edge, Config

# ----------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------

def load_image(file) -> np.ndarray:
    image = Image.open(file).convert("RGB")
    return np.array(image)


def detect_and_embed_faces(image: np.ndarray, detector="opencv", model="ArcFace"):
    results = []

    try:
        objs = DeepFace.extract_faces(
            img_path=image,
            detector_backend=detector,
            enforce_detection=False,
            align=True
        )
    except:
        return results

    for obj in objs:
        area = obj["facial_area"]
        x, y, w, h = area["x"], area["y"], area["w"], area["h"]

        face_img = image[y:y+h, x:x+w]

        rep = DeepFace.represent(
            img_path=face_img,
            model_name=model,
            detector_backend="skip",
            enforce_detection=False,
            align=False
        )

        embedding = np.array(rep[0]["embedding"])
        loc = (y, x + w, y + h, x)

        results.append({
            "embedding": embedding,
            "location": loc,
            "face_img": face_img
        })

    return results


def cluster_by_threshold(embeddings, threshold: float):
    n = len(embeddings)
    clusters = [-1] * n
    cluster_id = 0

    for i in range(n):
        if clusters[i] != -1:
            continue

        clusters[i] = cluster_id
        for j in range(i + 1, n):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            if dist < threshold:
                clusters[j] = cluster_id

        cluster_id += 1

    return np.array(clusters)


# ----------------------------------------------------------
# Image-to-Base64 Conversion
# ----------------------------------------------------------

def to_base64(face_img):
    if face_img.dtype != np.uint8:
        face_img = face_img.astype("uint8")
    if face_img.ndim == 2:
        face_img = np.stack([face_img] * 3, axis=-1)

    pil = Image.fromarray(face_img, mode="RGB")
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


# ----------------------------------------------------------
# Graph Builder (Cluster → Faces)
# ----------------------------------------------------------

def build_cluster_agraph(embeddings, labels, face_images, graph_threshold):
    nodes = []
    edges = []

    # Count faces in each cluster
    unique_clusters = sorted(set(labels))
    cluster_sizes = {cid: list(labels).count(cid) for cid in unique_clusters}

    # Cluster parent nodes (only where size > 1)
    for cid in unique_clusters:
        if cluster_sizes[cid] > 1:
            nodes.append(
                Node(
                    id=f"cluster_{cid}",
                    label=f"Cluster {cid}",
                    shape="dot",
                    size=65,
                    color="#00aaff",
                    font={"color": "white"},
                    title=f"Cluster {cid}"
                )
            )

    # Face nodes
    for idx, (label, face) in enumerate(zip(labels, face_images)):
        img_b64 = to_base64(face)

        nodes.append(
            Node(
                id=f"face_{idx}",
                label=f"Face {idx}",
                shape="circularImage",
                image=img_b64,
                brokenImage=img_b64,
                size=40,
                title=f"Face {idx} | Cluster {label}"
            )
        )

        # Connect ONLY IF cluster has 2+ faces
        if cluster_sizes[label] > 1:
            edges.append(
                Edge(
                    source=f"face_{idx}",
                    target=f"cluster_{label}",
                    title=f"in cluster {label}"
                )
            )

    # OPTIONAL similarity edges
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            if dist < graph_threshold:
                edges.append(
                    Edge(
                        source=f"face_{i}",
                        target=f"face_{j}",
                        title=f"{dist:.2f}",
                        color="rgba(200,200,200,0.4)"
                    )
                )

    config = Config(
            width="100%",
            height=720,
        directed=True,
        physics=True,
        hierarchical=False
    )

    return agraph(nodes=nodes, edges=edges, config=config)

# ----------------------------------------------------------
# Streamlit App (2 Column Layout)
# ----------------------------------------------------------

def main():
    st.set_page_config(layout="wide")

    st.title("Face Clustering + Interactive Graph")

    col_left, col_right = st.columns([0.35, 0.65])

    # ------------------------------------------------------
    # LEFT: Upload + sliders + face previews
    # ------------------------------------------------------
    with col_left:

        uploaded_files = st.file_uploader(
            "Upload images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        cluster_threshold = st.slider(
            "Clustering threshold",
            0.3, 1.5, 0.75,
            help="Lower = stricter grouping. Faces must be closer to belong to the same cluster."
        )

        graph_threshold = st.slider(
            "Graph similarity threshold",
            0.3, 1.5, 0.9,
            help="Faces more similar than this value are connected."
        )

        if not uploaded_files:
            st.info("Upload some images to continue.")
            return

        all_embeddings = []
        face_images = []
        faces_per_image = []

        st.subheader("Detected Faces Preview")

        # CSS (graph + scroll panel)

        # 🟦 1️⃣ FIRST: process all images (detect + embed)
        all_embeddings = []
        face_images = []
        faces_per_image = []

        for file in uploaded_files:
            img = load_image(file)
            faces = detect_and_embed_faces(img)

            if len(faces) == 0:
                st.warning(f"No faces detected in {file.name}")
                continue

            embeddings = [f["embedding"] for f in faces]
            crops = [f["face_img"] for f in faces]

            all_embeddings.extend(embeddings)
            face_images.extend(crops)

            faces_per_image.append({
                "name": file.name,
                "faces": faces,
                "embeddings": embeddings
            })


        # 🟦 2️⃣ If no faces → warn
        if len(face_images) == 0:
            st.info("No faces detected.")
            return

        # 🟦 3️⃣ NOW render the scroll panel

        for i, face in enumerate(face_images):
            col1, col2 = st.columns([0.3, 0.7])

            with col1:
                st.image(face, width=120)

            with col2:
                st.markdown(f"**Face {i}**")
                st.caption("Extracted from uploaded images")



        # 🟦 4️⃣ Not enough faces → stop here
        if len(all_embeddings) <= 1:
            st.warning("Not enough faces for clustering.")
            return


        all_embeddings = np.array(all_embeddings)
        labels = cluster_by_threshold(all_embeddings, cluster_threshold)

    # ------------------------------------------------------
    # RIGHT: Graph
    # ------------------------------------------------------
    with col_right:
        st.subheader("Interactive Graph")

        build_cluster_agraph(all_embeddings, labels, face_images, graph_threshold)


# Run
if __name__ == "__main__":
    main()
