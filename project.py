import numpy as np
from scipy.fft import fft2, ifft2  # Use fft2 and ifft2 from scipy.fft
from PIL import Image, ImageTk
from tkinter import ttk
from tkinter import filedialog
from tkinter import *
from PIL import Image
import cv2


# fourier_shift, fourier_oplan

# Create the main window
root = Tk()
root.title("Image Filter App")
root.geometry("500x300")  # Set window size (optional)
# Create a button
uploadButton = Button(root, text="Upload an image", fg="white", bg="green")
applyButton = Button(root, text="Apply filter", fg="black", bg="skyblue")
# Pack the button (or use another layout manager)
uploadButton.pack(pady=5)
applyButton.pack(pady=5)
# Create combobox
filters = ["Select a filter", "LPF", "HPF", "Mean Filter", "Median Filter", "Robert Edge Detector", "Prewitt Edge Detector",
           "Sobel Edge Detector", "Errosion", "Dilation", "Open", "Close", "Hough Transform", "Thresholding Segmentation"]
combo = ttk.Combobox(root, values=filters, state="readonly")
combo.current(0)
combo.pack(pady=5)

# Create image placeholders
oImage = Label(root, width=500, height=500,
               bg="lightgray", text="Original Image")
fImage = Label(root, width=500, height=500,
               bg="lightgray", text="Filtered Image")
oImage.pack(side=LEFT, padx=10, pady=10)
fImage.pack(side=RIGHT, padx=10, pady=10)

#######################################################################################################################################################################
img = None
arrayImg = None
# Upload Image Function
image_path = None  # Global variable to store the image path


def update_image():
    global image_path
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.gif")]
    )

    if image_path:
        try:
            # Open the image using PIL
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(
                image, (2000, 2000), interpolation=cv2.INTER_AREA)
            print(type(resized_image))
            print(resized_image.shape)
            global arrayImg
            arrayImg = resized_image
            image = Image.fromarray(resized_image)
            image = image.resize((500, 500), Image.LANCZOS)
            image = ImageTk.PhotoImage(image)

            global img
            img = image
            oImage.config(image=image)

        except (FileNotFoundError, PermissionError) as e:
            print(f"Error loading image: {e}")


# Set upload function to the upload button
uploadButton.config(command=update_image)

##################################################################################################################################################################
######################################################## Filters Functions#########################################################################################
fImg = None


def low():
    print(type(arrayImg))
    print(arrayImg.shape)
    global fImg
    fImg = cv2.GaussianBlur(arrayImg, (9, 9), 10, 10)
    print(type(fImg))
    print(fImg.shape)
    resized_image = cv2.resize(
        fImg, (2000, 2000), interpolation=cv2.INTER_AREA)
    fImg = Image.fromarray(resized_image)
    fImg = fImg.resize((500, 500), Image.LANCZOS)
    fImg = ImageTk.PhotoImage(fImg)
    print(type(fImg))
    # fImage.config(image=fImg, width=500, height=500)
    return fImg


def high():
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    global fImg
    fImg = cv2.filter2D(arrayImg, -1, kernel)
    resized_image = cv2.resize(
        fImg, (2000, 2000), interpolation=cv2.INTER_AREA)
    fImg = Image.fromarray(resized_image)
    fImg = fImg.resize((500, 500), Image.LANCZOS)
    fImg = ImageTk.PhotoImage(fImg)
    return fImg


def mean():
    global fImg
    fImg = cv2.blur(arrayImg, (9, 9))
    resized_image = cv2.resize(
        fImg, (2000, 2000), interpolation=cv2.INTER_AREA)
    fImg = Image.fromarray(resized_image)
    fImg = fImg.resize((500, 500), Image.LANCZOS)
    fImg = ImageTk.PhotoImage(fImg)
    return fImg


def median():
    global fImg
    fImg = cv2.medianBlur(arrayImg, 5)
    resized_image = cv2.resize(
        fImg, (2000, 2000), interpolation=cv2.INTER_AREA)
    fImg = Image.fromarray(resized_image)
    fImg = fImg.resize((500, 500), Image.LANCZOS)
    fImg = ImageTk.PhotoImage(fImg)
    return fImg


def robert():
    global fImg
    fImg = cv2.Canny(arrayImg, 100, 200)
    resized_image = cv2.resize(
        fImg, (2000, 2000), interpolation=cv2.INTER_AREA)
    fImg = Image.fromarray(resized_image)
    fImg = fImg.resize((500, 500), Image.LANCZOS)
    fImg = ImageTk.PhotoImage(fImg)
    return fImg


def prewitt():
    gray_image = cv2.cvtColor(arrayImg, cv2.COLOR_BGR2GRAY)
    prewitt_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    prewitt_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    prewitt_image = np.sqrt(prewitt_x**2 + prewitt_y**2)
    prewitt_image = cv2.normalize(
        prewitt_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    global fImg
    fImg = prewitt_image
    resized_image = cv2.resize(
        fImg, (2000, 2000), interpolation=cv2.INTER_AREA)
    fImg = Image.fromarray(resized_image)
    fImg = fImg.resize((500, 500), Image.LANCZOS)
    fImg = ImageTk.PhotoImage(fImg)
    return fImg


def sobel():
    gray_image = cv2.cvtColor(arrayImg, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_image = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_image = cv2.normalize(
        sobel_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    global fImg
    fImg = sobel_image
    resized_image = cv2.resize(
        fImg, (2000, 2000), interpolation=cv2.INTER_AREA)
    fImg = Image.fromarray(resized_image)
    fImg = fImg.resize((500, 500), Image.LANCZOS)
    fImg = ImageTk.PhotoImage(fImg)
    return fImg


def errosion():
    kernel = np.ones((5, 5), np.uint8)
    global fImg
    fImg = cv2.erode(arrayImg, kernel=kernel, iterations=1)
    resized_image = cv2.resize(
        fImg, (2000, 2000), interpolation=cv2.INTER_AREA)
    fImg = Image.fromarray(resized_image)
    fImg = fImg.resize((500, 500), Image.LANCZOS)
    fImg = ImageTk.PhotoImage(fImg)
    return fImg


def dilation():
    kernel = np.ones((5, 5), np.uint8)
    global fImg
    fImg = cv2.dilate(arrayImg, kernel=kernel, iterations=1)
    resized_image = cv2.resize(
        fImg, (2000, 2000), interpolation=cv2.INTER_AREA)
    fImg = Image.fromarray(resized_image)
    fImg = fImg.resize((500, 500), Image.LANCZOS)
    fImg = ImageTk.PhotoImage(fImg)
    return fImg


def open():
    kernel = np.ones((5, 5), np.uint8)
    global fImg
    fImg = cv2.morphologyEx(arrayImg, cv2.MORPH_OPEN, kernel=kernel)
    resized_image = cv2.resize(
        fImg, (2000, 2000), interpolation=cv2.INTER_AREA)
    fImg = Image.fromarray(resized_image)
    fImg = fImg.resize((500, 500), Image.LANCZOS)
    fImg = ImageTk.PhotoImage(fImg)
    return fImg


def close():
    kernel = np.ones((5, 5), np.uint8)
    global fImg
    fImg = cv2.morphologyEx(arrayImg, cv2.MORPH_CLOSE, kernel=kernel)
    resized_image = cv2.resize(
        fImg, (2000, 2000), interpolation=cv2.INTER_AREA)
    fImg = Image.fromarray(resized_image)
    fImg = fImg.resize((500, 500), Image.LANCZOS)
    fImg = ImageTk.PhotoImage(fImg)
    return fImg


def hough():
    gray = cv2.cvtColor(arrayImg, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Set parameters for Hough circle detection
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=10, maxRadius=100)

    # Check if circles are detected

    # Convert circles from list to NumPy array
    circles = np.uint16(np.around(circles[0, :]))

    # Draw detected circles on the original image
    for i in circles:
        cv2.circle(arrayImg, (i[0], i[1]), i[2],
                   (0, 255, 0), 2)  # Draw the outer circle
        cv2.circle(arrayImg, (i[0], i[1]), 2, (0, 0, 255), 3)
    global fImg
    fImg = arrayImg
    resized_image = cv2.resize(
        fImg, (2000, 2000), interpolation=cv2.INTER_AREA)
    fImg = Image.fromarray(resized_image)
    fImg = fImg.resize((500, 500), Image.LANCZOS)
    fImg = ImageTk.PhotoImage(fImg)
    return fImg


def thresholding():
    gray = cv2.cvtColor(arrayImg, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    global fImg
    fImg = thresh
    resized_image = cv2.resize(
        fImg, (2000, 2000), interpolation=cv2.INTER_AREA)
    fImg = Image.fromarray(resized_image)
    fImg = fImg.resize((500, 500), Image.LANCZOS)
    fImg = ImageTk.PhotoImage(fImg)
    return fImg

#########################################################################################################################################################################


def LPF():
    fImage.config(image=low(), width=500, height=500)


def HPF():
    fImage.config(image=high(), width=500, height=500)


def MEAN():
    fImage.config(image=mean(), width=500, height=500)


def MEDIAN():
    fImage.config(image=median(), width=500, height=500)


def ROBERT():
    fImage.config(image=robert(), width=500, height=500)


def PREWITT():
    fImage.config(image=prewitt(), width=500, height=500)


def SOBEL():
    fImage.config(image=sobel(), width=500, height=500)


def ERROSION():
    fImage.config(image=errosion(), width=500, height=500)


def DILATION():
    fImage.config(image=dilation(), width=500, height=500)


def OPEN():
    fImage.config(image=open(), width=500, height=500)


def CLOSE():
    fImage.config(image=close(), width=500, height=500)


def HOUGH():
    fImage.config(image=hough(), width=500, height=500)


def THRESHOLD():
    fImage.config(image=thresholding(), width=500, height=500)


def apply():
    selection = combo.get()
    if selection == "LPF":
        print("You have choosed LPF")
        LPF()
    elif selection == "HPF":
        print("You have choosed HPF")
        HPF()
    elif selection == "Mean Filter":
        print("You have choosed Mean Filter")
        MEAN()
    elif selection == "Median Filter":
        print("You have choosed median filter")
        MEDIAN()
    elif selection == "Robert Edge Detector":
        print("You have choosed Robert Edge Detector")
        ROBERT()
    elif selection == "Prewitt Edge Detector":
        print("You have choosed prewitt edge detector")
        PREWITT()
    elif selection == "Sobel Edge Detector":
        print("You have choosed sobel edge detector")
        SOBEL()
    elif selection == "Errosion":
        print("You have choosed errosion")
        ERROSION()
    elif selection == "Dilation":
        print("You have choosed dilation")
        DILATION()
    elif selection == "Open":
        print("You have choosed open")
        OPEN()
    elif selection == "Close":
        print("You have choosed close")
        CLOSE()
    elif selection == "Hough Transform":
        print("You have choosed hough transform")
        HOUGH()
    elif selection == "Thresholding Segmentation":
        print("You have choosed thresholding segmentation")
        THRESHOLD()
    else:
        print("No filter choosen")


# Apply filter function
applyButton.config(command=apply)


root.mainloop()
