import os
import time
import json
from datetime import timedelta
from os import listdir

import face_recognition as fr
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip

SAVING_FRAMES_PER_SECOND = 1


def format_timedelta(td):
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")

    ms = round(int(ms) / 10000)
    return f"{result}.{ms:02}".replace(":", "-")


def from_video_to_frames(video_file):
    video_clip = VideoFileClip(video_file)
    filename, _ = os.path.splitext(video_file)

    if not os.path.isdir(filename):
        os.mkdir(filename)

    saving_frames_per_second = min(video_clip.fps, SAVING_FRAMES_PER_SECOND)
    step = 1 / video_clip.fps if saving_frames_per_second == 0 else 1 / saving_frames_per_second

    for current_duration in np.arange(0, video_clip.duration, step):
        frame_duration_formatted = format_timedelta(timedelta(seconds=current_duration)).replace(":", "-")
        frame_filename = os.path.join("D:/PROGRAMMING/py_face_rec/temporary_files/",
                                      f"frame{frame_duration_formatted}.jpg")

        video_clip.save_frame(frame_filename, current_duration)


def loadImages():
    # return array of images

    path = "D:/PROGRAMMING/py_face_rec/temporary_files/"
    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = fr.load_image_file(path + image)
        loadedImages.append(img)
    return loadedImages


def remove_images():
    path = "D:/PROGRAMMING/py_face_rec/temporary_files/"
    imagesList = listdir(path)
    for image in imagesList:
        os.remove(path + image)


def analyze():
    video_file = input()
    from_video_to_frames(video_file)
    frames = loadImages()

    with open("database.json") as db:
        persons = json.load(db)

    vals = persons.values()
    if len(vals) == 0:
        print("Database is empty!")
        return

    for val in vals:
        np_val = np.array(val)
        compares = []
        for frame in frames:
            frame_enc = fr.face_encodings(frame)
            if len(frame_enc) == 0:
                continue
            compare = fr.compare_faces(np_val, frame_enc)[0]
            compares.append(compare)
        print(compares)
        yes = False
        for comp in compares:
            if comp:
                yes = True
        key = []
        for key in persons.keys():
            if persons[key] == val:
                print(key, "is in this video")
                continue
        if yes:
            print(key, "is in this video")
        else:
            print(key, "isn't in this video")

    remove_images()


def add_photo():
    photo_file = input()
    name = input()
    with open("database.json") as db:
        persons = json.load(db)

    img = fr.load_image_file(photo_file)
    img_encodings = fr.face_encodings(img)
    img_encodings = list(img_encodings[0])
    #persons[name] = img_encodings
    persons.setdefault(name, img_encodings)

    with open("database.json", 'w') as db:
        json.dump(persons, db)


def clear_database():
    with open("database.json", 'w') as db:
        json.dump({}, db)


def main():
    while (True):
        number = int(input("""
Enter 1, if you want to analyze video file 
Enter 2, if you want to add new photo(person) to database
Enter 3, if you want to clear database
Enter other integer to exit from program\n"""))
        if number == 1:
            analyze()
        elif number == 2:
            add_photo()
        elif number == 3:
            clear_database()
        else:
            break
    print("The end of the program")


if __name__ == '__main__':
    main()
