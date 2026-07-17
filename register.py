import cv2
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# DB setup
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker

engine = create_engine("sqlite:///attendance.db", echo=False)
Base = declarative_base()
Session = sessionmaker(bind=engine)

class Person(Base):
    __tablename__ = "persons"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    embedding = Column(Text)

Base.metadata.create_all(engine)

session = Session()

name = input("Enter name: ")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

cap = cv2.VideoCapture(0)

print("Look at the camera...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)

    if face is not None:
        if face.ndim == 3:
            face = face.unsqueeze(0)

        emb = resnet(face).detach().cpu().numpy().flatten()
        emb_str = ",".join(map(str, emb))

        person = Person(name=name, embedding=emb_str)
        session.add(person)
        session.commit()

        print(f"{name} registered successfully!")
        break

    cv2.imshow("Register Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
session.close()