import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

for i in range(1, 26):
    img_path = rf'C:\Users\prime\OneDrive\Desktop\Face-Detection-Recognition-Using-OpenCV-in-Python-master\Face-Detection-Recognition-Using-OpenCV-in-Python-master\attachments\i ({i}).tif'
    img = cv2.imread(img_path)

    if img is None:
        continue

    img = cv2.flip(img, -1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 1.1, 4, minSize=(50, 50)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # ---- usable face region (remove forehead) ----
        uf_y = y + int(0.25 * h)
        uf_h = int(0.65 * h)

        # ---- geometric landmarks ----
        left_eye  = (x + int(0.35 * w), uf_y + int(0.45 * uf_h))
        right_eye = (x + int(0.65 * w), uf_y + int(0.45 * uf_h))
        nose      = (x + int(0.50 * w), uf_y + int(0.75 * uf_h))
        mouth     = (x + int(0.50 * w), uf_y + int(0.95 * uf_h))

        for pt in [left_eye, right_eye, nose, mouth]:
            cv2.circle(img, pt, 2, (0, 0, 255), -1)

    cv2.imshow("Geometric Facial Points", img)
    if cv2.waitKey(0) & 0xFF == 27:
        break

cv2.destroyAllWindows()
