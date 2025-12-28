#Implementation 1 & 2
"""
import pickle
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

image_path = r"elijah eberhard test.jpg" #CHANGE TO SOMETHING ELSE
CLASSIFIER_PATH = "face_classifier1.pkl" #only student faces

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

mtcnn = MTCNN(keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

with open(CLASSIFIER_PATH, "rb") as f:
    model, out_encoder, in_encoder = pickle.load(f)

def preprocess(image):
    face = mtcnn(image)
    if face is None:
        return None
    return face.unsqueeze(0).to(device)

def get_embedding(tensor):
    with torch.no_grad():
        return facenet(tensor).cpu().numpy()[0]

def recognize_best_match(image_path):
    print("loading",image_path)
    image = Image.open(image_path).convert("RGB")

    tensor = preprocess(image)
    if tensor is None:
        print("no face found")
        return

    embedding = get_embedding(tensor)
    embedding_normalized = in_encoder.transform([embedding])

    distances = model.kneighbors(embedding_normalized, n_neighbors=5, return_distance=True)[0]
    indices = model.kneighbors(embedding_normalized, n_neighbors=5, return_distance=True)[1]

    best_index = indices[0][0]
    best_distance = distances[0][0]
    best_label = out_encoder.inverse_transform([model._y[best_index]])[0]

    print("best match:",best_label)
    print("distance:",best_distance)

recognize_best_match(image_path)
"""


#Implementation 3
#"""
import pickle
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1


image_path = r"jim carrey test.jpg"  #CHANGE TO SOMETHING ELSE
CLASSIFIER_PATH = "face_classifier2.pkl" #includes celebrity faces

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

mtcnn = MTCNN(keep_all=False, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


with open(CLASSIFIER_PATH, "rb") as f:
    model, out_encoder, in_encoder = pickle.load(f)


def preprocess(image):
    face = mtcnn(image)
    if face is None:
        return None
    return face.unsqueeze(0).to(device)


def get_embedding(tensor):
    with torch.no_grad():
        return facenet(tensor).cpu().numpy()[0]


def recognize_top5(image_path):
    print("testing",image_path)
    image = Image.open(image_path).convert("RGB")

    tensor = preprocess(image)
    if tensor is None:
        print("no face found")
        return

    embedding = get_embedding(tensor)
    embedding_normalized = in_encoder.transform([embedding])

    distances = model.kneighbors(embedding_normalized, n_neighbors=5, return_distance=True)[0]
    indices = model.kneighbors(embedding_normalized, n_neighbors=5, return_distance=True)[1]

    print("top 5 matches:")
    for i in range(5):
        distance = distances[0][i]
        index = indices[0][i]
        label = out_encoder.inverse_transform([model._y[index]])[0]
        print((i+1),label,"distance:",distance)

recognize_top5(image_path)
#"""


#ON LAPTOP
#C:\Users\Elija\pycharm_venvs\CSCI335AIFinalProject\Scripts\Activate.ps1
#cd "C:\Users\Elija\OneDrive - Hendrix College\College Stuff\Sophomore 1st Semester\AI\Final Project\Program"
#python FaceRecognizer.py