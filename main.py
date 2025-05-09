import sys
from PIL import Image
import trimesh
from rembg import remove
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import cv2
import numpy as np

TEXT_MODE = "text"
IMAGE_MODE = "image"

def get_user_input():
    choice = input("Type 'text' to enter a prompt or 'image' to upload a photo: ").strip().lower()
    if choice == TEXT_MODE:
        return TEXT_MODE, input("Enter a short description (e.g., 'a toy car'): ")
    elif choice == IMAGE_MODE:
        image_path = input("Enter the path to the image file (e.g., toy.png): ").strip()
        return IMAGE_MODE, image_path
    else:
        print("Invalid input")
        sys.exit()

def generate_text_3d_model(prompt):
    print(f"[Mock] Generating 3D model for text: {prompt}")
    mesh = trimesh.creation.icosphere()
    mesh.export("text_output.obj")
    return mesh

def estimate_depth(image_path):
    print("[INFO] Loading MiDaS model...")
    model_type = "MiDaS_small"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device).eval()
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    print("[INFO] Reading image...")
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()
    return depth

def depth_to_mesh(depth, scale=0.005):
    h, w = depth.shape
    print(f"[INFO] Converting depth map ({h}x{w}) to 3D mesh...")
    vertices = []
    faces = []

    for y in range(h):
        for x in range(w):
            z = depth[y, x] * scale
            vertices.append([x, y, z])

    for y in range(h - 1):
        for x in range(w - 1):
            i = y * w + x
            faces.append([i, i + 1, i + w])
            faces.append([i + 1, i + 1 + w, i + w])

    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
    mesh.export("image_output.obj")
    print("[INFO] Saved mesh to 'image_output.obj'")
    return mesh

def visualize_with_matplotlib(mesh):
    vertices = mesh.vertices
    faces = mesh.faces

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for face in faces:
        tri = vertices[face]
        x, y, z = zip(*tri)
        x += (x[0],)
        y += (y[0],)
        z += (z[0],)
        ax.plot(x, y, z, color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("3D Model (Wireframe View)")
    plt.show()

if __name__ == "__main__":
    mode, data = get_user_input()

    if mode == TEXT_MODE:
        model = generate_text_3d_model(data)
        print("Saved 3D model to text_output.obj")
    else:
        print("[INFO] Generating depth and mesh from image...")
        depth = estimate_depth(data)
        model = depth_to_mesh(depth)
        print("Saved 3D model to image_output.obj")

    visualize_with_matplotlib(model)
