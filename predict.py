from PIL import Image

from facenet import Facenet

if __name__ == "__main__":
    model = Facenet()

    image_1 = "img/1.jpg"
    try:
        image_1 = Image.open(image_1)
    except:
        print('Image_1 Open Error! Try again!')

    image_2 = "img/1h.jpg"
    try:
        image_2 = Image.open(image_2)
    except:
        print('Image_2 Open Error! Try again!')

    probability = model.detect_image(image_1, image_2, show=True)
    print(probability)
