import cv2

def iterate_cameras():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)

        if not cap.read()[0]:
            break

        else :
            print(f"Camera {index} is available")
            arr.append(index)
            cap.release()
            index += 1

    return arr

if __name__ == "__main__":
    print(iterate_cameras())