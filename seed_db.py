import cv2
import numpy as np
import os
import sqlite3
from database import init_db, get_connection
from vision.recognizer import FaceRecognizer

def seed_database():
    init_db()
    conn = get_connection()
    cursor = conn.cursor()

    default_students = [
        {'id': 'S001', 'name': 'Ravi Kumar', 'class': 'CSE - III Year'},
        {'id': 'S002', 'name': 'Sumanth', 'class': 'CSE - III Year'},
        {'id': 'S003', 'name': 'Kiran', 'class': 'CSE - III Year'},
        {'id': 'S004', 'name': 'Priya Sharma', 'class': 'CSE - III Year'},
        {'id': 'S005', 'name': 'Ananya Roy', 'class': 'CSE - III Year'}
    ]

    for s in default_students:
        cursor.execute('''
            INSERT OR REPLACE INTO students (student_id, name, class_name)
            VALUES (?, ?, ?)
        ''', (s['id'], s['name'], s['class']))
        
    conn.commit()
    conn.close()

    recognizer = FaceRecognizer()
    for idx, s in enumerate(default_students):
        # Create distinct normalized face images for each student
        face_img = np.full((120, 120, 3), (200 + idx*5, 210 - idx*8, 220 + idx*4), dtype=np.uint8)
        cv2.circle(face_img, (35 + idx*3, 45), 6, (30, 40, 50), -1)
        cv2.circle(face_img, (75 + idx*3, 45), 6, (30, 40, 50), -1)
        cv2.line(face_img, (40 + idx*2, 85), (80 - idx*2, 85), (50, 60, 75), 2)

        recognizer.register_student_face_samples(s['id'], [face_img])

    print("Database seeded with distinct registered student encodings.")

if __name__ == "__main__":
    seed_database()
