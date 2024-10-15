import sys
import os
import numpy as np
import cv2
from flask import Flask, flash, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
app = Flask(__name__, static_url_path='')
app.secret_key = os.urandom(24)

app.config['CARTOON_FOLDER'] = 'cartoon_images'
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/uploads/<filename>')
def upload_img(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/cartoon_images/<filename>')
def cartoon_img(filename):
    return send_from_directory(app.config['CARTOON_FOLDER'], filename)


# Cartoonization Style 1 (K-means and Adaptive Threshold with Original Colors)
def cartoonize_1(img, k):
    # Convert the input image to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform adaptive threshold to get edges
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)

    # Defining input data for clustering
    data = np.float32(img).reshape((-1, 3))

    # Defining criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    # Applying kmeans
    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)

    # Reshape the output data to the size of input image
    result = center[label.flatten()]
    result = result.reshape(img.shape)

    # Smooth the result to reduce noise
    blurred = cv2.medianBlur(result, 5)

    # Combine the result and edges to get final cartoon effect with original colors
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

    # Instead of combining with the edges, we can directly blend the edges with the original image
    # This way, we can keep the real colors intact.
    cartoon = cv2.addWeighted(img, 0.5, cartoon, 0.5, 0)

    return cartoon


def cartoonize_2_watercolor(img):
    # Step 1: Apply a bilateral filter to reduce noise and maintain edge clarity
    img_bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 2: Convert the image to grayscale and use adaptive thresholding to find edges
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, blockSize=9, C=2)

    # Step 3: Perform color quantization to create flat regions of color
    Z = img_bilateral.reshape((-1, 3))
    Z = np.float32(Z)

    # Define criteria and apply kmeans to quantize colors
    K = 8  # Number of color clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back into an image with quantized colors
    center = np.uint8(center)
    img_quantized = center[label.flatten()]
    img_quantized = img_quantized.reshape(img.shape)

    # Step 4: Convert edges to 3 channels to match quantized image's shape
    img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)

    # Step 5: Blend the quantized image and edges for a soft watercolor effect
    img_watercolor = cv2.addWeighted(img_quantized, 0.7, img_edges, 0.3, 0)

    # Step 6: Apply a Gaussian blur to soften the image further
    img_blurred = cv2.GaussianBlur(img_watercolor, (7, 7), sigmaX=0, sigmaY=0)

    # Step 7: Blend the original image with the watercolor effect to retain colors
    img_final = cv2.addWeighted(img_blurred, 0.6, img, 0.4, 0)  # Adjust weights as needed

    return img_final


# Cartoonization Style 3 (Pencil Sketch with Color Tint)
def cartoonize_3(img):
    # Convert the image to gray scale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Use Gaussian Blur to reduce noise
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
    # Use adaptive thresholding to get clear edges
    edges = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, blockSize=9, C=2)

    # Create a pencil sketch in gray
    imout_gray, _ = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    
    # Combine the edges with the gray pencil sketch to maintain sharp outlines
    clarified_sketch = cv2.bitwise_and(imout_gray, imout_gray, mask=edges)

    return clarified_sketch






# Cartoonization Style 4 (Pencil Sketch Colored)
def cartoonize_4(img):
    # Pencil sketch of image
    imout_gray, imout_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    
    return imout_color  


def cartoonize_6(img):
    # Step 1: Apply detail enhancement to bring out fine details
    dst = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

    # Step 2: Apply bilateral filter to smooth the image while keeping edges sharp
    dst = cv2.bilateralFilter(dst, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 3: Adjust brightness and contrast for a more vivid appearance
    dst = cv2.convertScaleAbs(dst, alpha=1.3, beta=30)  # Increase contrast and brightness

    # Step 4: Optional - Sharpen the image further to enhance HD quality
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])  # Sharpening kernel
    dst = cv2.filter2D(dst, -1, kernel)

    return dst



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        style = request.form.get('style')
        print(style)

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        file_name = os.path.basename(file_path)

        # Reading the uploaded image
        img = cv2.imread(file_path)

        # Apply selected cartoon style
        if style == "Style1":
            cart_fname = file_name + "_style1_cartoon.jpg"
            cartoonized = cartoonize_1(img, 8)
        elif style == "Style2":
            cart_fname = file_name + "_style2_cartoon.jpg"
            cartoonized = cartoonize_2_watercolor(img)
        elif style == "Style3":
            cart_fname = file_name + "_style3_cartoon.jpg"
            cartoonized = cartoonize_3(img)
        elif style == "Style4":
            cart_fname = file_name + "_style4_cartoon.jpg"
            cartoonized = cartoonize_4(img)
        elif style == "Style6":
            cart_fname = file_name + "_style6_cartoon.jpg"
            cartoonized = cartoonize_6(img)
        else:
            flash('Please select a style')
            return render_template('index.html')

        # Save the cartoonized image to ./cartoon_images
        cartoon_path = os.path.join(basepath, 'cartoon_images', secure_filename(cart_fname))
        fname = os.path.basename(cartoon_path)
        cv2.imwrite(cartoon_path, cartoonized)

        return render_template('predict.html', file_name=file_name, cartoon_file=fname)

    return ""


if __name__ == '__main__':
    app.run(debug=True, host="localhost", port=8080)
