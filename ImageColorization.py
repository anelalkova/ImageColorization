import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Global variables to store image paths and colorized image
img_path = None
original_img = None
colorized_img = None


# Function to colorize the selected black-and-white image
def colorize_img(image_path):
    prototxt_path = 'model/colorization_deploy_v2.prototxt'
    model_path = 'model/colorization_release_v2.caffemodel'
    kernel_path = 'model/pts_in_hull.npy'

    # Load the pre-trained network model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Load cluster points for colorization
    points = np.load(kernel_path)
    points = points.transpose().reshape(2, 313, 1, 1)

    # Set the model's colorization parameters
    net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Load and process the image
    bw_image = cv2.imread(image_path)
    if bw_image is None:
        return None  # Error: Unable to load the image

    normalized = bw_image.astype("float32") / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50  # Adjusting the L channel

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255.0 * colorized).astype("uint8")
    return colorized  # Return the colorized image


# Function to open file dialog and select an image from Downloads folder
def open_image():
    global img_path, original_img

    # Open file dialog in the user's Downloads folder
    downloads_folder = os.path.join(os.path.expanduser('~'), 'Downloads')
    img_path = filedialog.askopenfilename(initialdir=downloads_folder, title="Select an Image",
                                          filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("all files", "*.*")))
    if img_path:
        # Load and display the original image
        original_img = cv2.imread(img_path)
        if original_img is not None:
            display_image(original_img)  # Display the original image
            colorize_button.config(state=tk.NORMAL)  # Enable the colorize button


# Function to resize and display an image on the canvas
def display_image(image):
    # Resize the image to fit within the 600x400 canvas while maintaining aspect ratio
    h, w = image.shape[:2]
    canvas_w, canvas_h = 600, 400  # Define the canvas size

    # Calculate the aspect ratio to resize the image
    ratio_w = canvas_w / w
    ratio_h = canvas_h / h
    ratio = min(ratio_w, ratio_h)

    # Resize the image based on the smaller ratio to fit within the canvas
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    resized_img = cv2.resize(image, (new_w, new_h))

    # Convert the image from BGR to RGB for displaying
    image_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(image_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    # Clear the canvas before displaying a new image
    canvas.delete("all")

    # Center the image on the canvas
    canvas.create_image(canvas_w // 2, canvas_h // 2, anchor=tk.CENTER, image=img_tk)
    canvas.image = img_tk  # Keep a reference to avoid garbage collection


# Function to colorize the currently opened image
def colorize_image():
    global colorized_img

    if img_path:
        # Colorize the image
        colorized_img = colorize_img(img_path)
        if colorized_img is not None:
            display_image(colorized_img)  # Display the colorized image


# Function to save the colorized image
def save_image():
    if colorized_img is not None:
        # Create directory for saving the colorized image
        output_folder = r'C:\ImageColorization'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save the image with the same name and '_colorized' suffix
        image_name = os.path.basename(img_path).split('.')[0]
        save_path = os.path.join(output_folder, f'{image_name}_colorized.jpg')
        cv2.imwrite(save_path, colorized_img)
        messagebox.showinfo("Success", f"Colorized image saved as: {save_path}")
    else:
        messagebox.showerror("Error", "No image has been colorized to save.")


# Create the Tkinter window
window = tk.Tk()
window.title("Image Colorization")
window.geometry('600x600')

# Create a canvas to display the image in the center
canvas = tk.Canvas(window, width=600, height=400)
canvas.pack()

# Create buttons for "Open", "Colorize", "Save", and "Exit"
open_button = tk.Button(window, text="Open Image", command=open_image, width=20)
open_button.pack(pady=10)

colorize_button = tk.Button(window, text="Colorize Image", command=colorize_image, width=20, state=tk.DISABLED)
colorize_button.pack(pady=10)

save_button = tk.Button(window, text="Save Image", command=save_image, width=20)
save_button.pack(pady=10)

exit_button = tk.Button(window, text="Exit", command=window.quit, width=20)
exit_button.pack(pady=10)

# Run the Tkinter main loop
window.mainloop()
