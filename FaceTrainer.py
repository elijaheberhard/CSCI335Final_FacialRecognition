#Implementation 1, 2, & 3
#"""
import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.neighbors import KNeighborsClassifier
import pickle

dataset_path = r"C:\Users\Eberh\OneDrive - Hendrix College\College Stuff\Sophomore 1st Semester\AI\Final Project\Program\photos"
CLASSIFIER_PATH = "face_classifier.pkl"

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

detector = MTCNN(keep_all=False, device=device, margin=40, min_face_size=20, thresholds=[0.6, 0.7, 0.7])
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def preprocess(photo):
    tensor = detector(photo)
    if tensor is None:
        return None
    return tensor.unsqueeze(0).to(device)

def get_embedding(tensor):
    with torch.no_grad():
        return facenet_model(tensor).cpu().numpy()[0]

X, y = [], []

for item in os.listdir(dataset_path):
    item_path = os.path.join(dataset_path, item)
    if os.path.isdir(item_path):
        for filename in os.listdir(item_path):
            if not filename.lower().endswith('.jpg'):
                continue
            filepath = os.path.join(item_path, filename)

            try:
                photo = Image.open(filepath).convert('RGB')
            except:
                print("failed to open image")
                continue

            tensor = preprocess(photo)
            if tensor is None:
                print("no face found")
                continue

            embedding = get_embedding(tensor)
            X.append(embedding)
            y.append(os.path.splitext(filename)[0])

    elif item.lower().endswith('.jpg'):
        filepath = item_path
        print("processing file",filepath)

        try:
            photo = Image.open(filepath).convert('RGB')
        except:
            print("failed to open image")
            continue

        tensor = preprocess(photo)
        if tensor is None:
            print("no face found")
            continue

        embedding = get_embedding(tensor)
        X.append(embedding)
        Y.append(os.path.splitext(item)[0])

if not X:
    raise RuntimeError("no faces detected")

X = np.array(X)
Y = np.array(Y)
print("collected",len(X),"face embeddings")

encode_in = Normalizer(norm='l2')
X_encoded = encode_in.transform(X)

encode_out = LabelEncoder()
Y_encoded = encode_out.fit_transform(y)

model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
model.fit(X_encoded, Y_encoded)

with open(CLASSIFIER_PATH, 'wb') as f:
    pickle.dump((model, encode_out, encode_in), f)
print("done")

#"""

"""
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/elijaheberhard/CSCI335Final_FacialRecognition.git
git push -u origin main"""