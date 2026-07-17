import cv2
import numpy as np
import torch
import datetime
import traceback
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from deep_sort_realtime.deepsort_tracker import DeepSort

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker

# =========================
# DATABASE SETUP
# =========================
engine = create_engine("sqlite:///attendance.db", echo=False)
Base = declarative_base()
Session = sessionmaker(bind=engine)

class Person(Base):
    __tablename__ = "persons"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    embedding = Column(Text)

class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True)
    person_id = Column(Integer, ForeignKey("persons.id"))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(engine)

# =========================
# LOAD KNOWN FACES
# =========================
def load_known_faces(session):
    persons = session.query(Person).all()

    known_embeddings = []
    names = []
    ids = []

    for p in persons:
        emb = np.fromstring(p.embedding, sep=",")
        norm = np.linalg.norm(emb)

        if norm > 0:
            emb = emb / norm

        known_embeddings.append(emb)
        names.append(p.name)
        ids.append(p.id)

    if len(known_embeddings) == 0:
        return np.empty((0, 512)), names, ids

    return np.vstack(known_embeddings), names, ids

# =========================
# FACE RECOGNITION
# =========================
def recognize_face(embedding, known_embeddings):
    if len(known_embeddings) == 0:
        return None, 0.0

    embedding = embedding / np.linalg.norm(embedding)
    similarities = np.dot(known_embeddings, embedding)

    best_idx = np.argmax(similarities)
    best_sim = float(similarities[best_idx])

    return best_idx, best_sim

# =========================
# MAIN FUNCTION
# =========================
def main():
    session = Session()

    try:
        known_embeddings, names, person_ids = load_known_faces(session)
        print(f"Loaded {len(names)} known faces.")

        if len(names) == 0:
            print("No registered faces found. Run register.py first.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading models...")
        mtcnn = MTCNN(keep_all=True, device=device)
        resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        tracker = DeepSort(max_age=30)

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Camera not working")
            return

        attended_today = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(frame_rgb)

            face_embeddings = []
            detections = []

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = [int(v) for v in box]

                    face = frame_rgb[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    face_pil = Image.fromarray(face)
                    face_tensor = mtcnn(face_pil)

                    if face_tensor is not None:
                        if face_tensor.ndim == 3:
                            face_tensor = face_tensor.unsqueeze(0)

                        emb = resnet(face_tensor).detach().cpu().numpy().flatten()
                        face_embeddings.append(emb)

                        detections.append(([x1, y1, x2 - x1, y2 - y1], 1.0, "face"))

            tracks = tracker.update_tracks(detections, frame=frame)

            for i, track in enumerate(tracks):
                if not track.is_confirmed():
                    continue

                l, t, r, b = map(int, track.to_ltrb())

                if i < len(face_embeddings):
                    emb = face_embeddings[i]
                    person_idx, sim = recognize_face(emb, known_embeddings)

                    if person_idx is not None and sim > 0.5:
                        person_id = person_ids[person_idx]
                        name = names[person_idx]

                        label = f"{name} ({sim:.2f})"

                        if person_id not in attended_today:
                            attendance = Attendance(person_id=person_id)
                            session.add(attendance)
                            session.commit()
                            attended_today.add(person_id)

                            print(f"{name} marked present")

                        color = (0, 255, 0)
                    else:
                        label = "Unknown"
                        color = (0, 0, 255)
                else:
                    label = "Unknown"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (l, t), (r, b), color, 2)
                cv2.putText(frame, label, (l, t - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("Attendance System", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("System stopped.")

    except Exception as exc:
        print("Error:", exc)
        traceback.print_exc()

    finally:
        session.close()

# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()