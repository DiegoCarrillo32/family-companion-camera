import threading
import time
import requests
import cv2
from google.cloud import vision

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(0)

isTesting = False

url_post = "http://localhost:5000/detected_face"
# Detecta una cara encontrada en un frame de un video accesando a la camara
def detect_face_in_camera():
    while True:

        result, video_frame = video_capture.read()  # read frames from the video
        if result is False:
            break

        gray_image = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        # Si se detecta una cara ( Len > 1 ) Se toma una captura
        cv2.imshow(
            "My Face Detection Project", video_frame
        )  # display the processed frame in a window named "My Face Detection Project"

        if len(faces) >= 1 and isTesting is False:
            print("Cara detectada!")
            cv2.imwrite('faces/face2.jpg', video_frame)
            print("Consultando con Google Cloud")
            # A esta funcion meter paralelismo para que no se detenga el proceso cuando hace la llamada
            # detect_emotion('faces/face2.jpg')
            thread = threading.Thread(target=detect_emotion, args=('faces/face2.jpg',))
            thread.start()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("-- DETENIENDO EL SCRIPT --")
            break
    video_capture.release()
    cv2.destroyAllWindows()


# Detecta las emociones de una imagen dada por parametro por medio de Google Cloud AI
def detect_emotion(path):
    """Detects faces in an image."""
    global isTesting
    isTesting = True
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = (
        "UNKNOWN",
        "VERY_UNLIKELY",
        "UNLIKELY",
        "POSSIBLE",
        "LIKELY",
        "VERY_LIKELY",
    )
    print("Faces:")
    for face in faces:
        print('------------------- PERSONA -------------------------')
        print(f"anger: {likelihood_name[face.anger_likelihood]}")
        print(f"joy: {likelihood_name[face.joy_likelihood]}")
        print(f"surprise: {likelihood_name[face.surprise_likelihood]}")
        print(f"sorrow: {likelihood_name[face.sorrow_likelihood]}")
        print('------------------- END -------------------------')
        vertices = [
            f"({vertex.x},{vertex.y})" for vertex in face.bounding_poly.vertices
        ]

        print("face bounds: {}".format(",".join(vertices)))
    data = {
        "anger": likelihood_name[faces[0].anger_likelihood],
        "joy": likelihood_name[faces[0].joy_likelihood],
        "surprise": likelihood_name[faces[0].surprise_likelihood],
        "sorrow": likelihood_name[faces[0].sorrow_likelihood],
        "family_user_chatId": "6056556009"
    }

    post_response = requests.post(url_post, json=data)
    post_json = post_response.json()
    print(post_json)
    time.sleep(5)
    isTesting = False

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )


detect_face_in_camera()
