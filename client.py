import socket
import time
import streamlit as st
from PIL import Image
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import google.generativeai as genai
import pandas as pd

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel("gemini-pro-vision")

## Function to load gemini model and gemini pro and get responses
def get_gemini_response(image, pred_class, pred_prob, hum_temp):
    response = model.generate_content([f"""You are a plant care expert look at the plant and analyze the data to suggest recommendations for plant care. 
                                       This image is passed through our highly accurate AI model
                                       which will provide the prediction class and it's probability: 
                                       Predicition Class: {pred_class}, Confidence: {pred_prob}%. Also use the live data from
                                       temperature and humidity sensor: {hum_temp} to suggest some plant care. Analyze this data along with image and generate a report of the plant""",image])
    return response.text


def receive_image(user_input):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
       
        s.connect(("raspberrypi", 5000))
        s.send(bytes(user_input, "utf-8"))
        if user_input.strip() == '1':
            data = s.recv(1024)
            with open("received_image.jpg", "wb") as f:
                while data:
                    f.write(data)
                    data = s.recv(1024)
            print("Image received and saved.")

def receive_data(user_input):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(("raspberrypi", 5000))
        s.send(bytes(user_input, "utf-8"))
        data = s.recv(1024)
        result = data.decode("utf-8")
        return result


def display_image():
    image_path = "received_image.jpg"
    if os.path.exists(image_path):
        image = Image.open(image_path)
        st.image(image, caption='Received Image', width=300)
    else:
        st.warning("No image available. Please receive an image first.")


def predict_class(model, class_names, img):
    img  = load_and_prep_image(img, scale=False)  # don't need to scale for EfficientNetb0
    pred_prob = model.predict(tf.expand_dims(img, axis=0))   # make prediction on image with shape [1, 224, 224, 3] (same shape on which our model waas trained on)
    pred_class = class_names[pred_prob.argmax()]  # get the index with the highest prediction probability

    return pred_class, pred_prob
    

# Load and prep Image for Image Processing
# Create a function to load and prepare images
def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  specified shape (img_shape, img_shape, color_channels=3).

  Args:
    filename (str): path to target image
    img_shape (int): height/width dimension of target image size
    scale (bool): scale pixel values from 0-255 to 0-1 or not.

  Returns:
    Image tensor of shape (img_shape, img_shape, 3)
  """

  # Read in the image
  img = tf.io.read_file(filename)

  # Decode image into tensor
  img = tf.io.decode_image(img, channels=3)

  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])

  # Scale? yes/no
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.

  else:
    return img # don't need to rescale for EfficicentNet models in Tensorflow


# Function to extract and return temperature and humidity
def temp_humidity(line):
    # Split the line by spaces to extract temperature and humidity
    parts = line.split()
    
    # Find the parts containing temperature and humidity
    temp_part = next((part for part in parts if 'Temp=' in part), None)
    humid_part = next((part for part in parts if 'Humidity=' in part), None)
    
    # Extract the numerical values from the parts
    temperature = float(temp_part.split('=')[1][:-1])  # Remove 'C' from temperature
    humidity = float(humid_part.split('=')[1][:-1])   # Remove '%' from humidity
    return temperature, humidity




# Function to check if temperature and humidity are within normal range
def check_status(temperature, humidity):
    if temperature >= 20 and temperature <= 25 and humidity >= 40 and humidity <= 60:
        return "Normal"
    else:
        return "Abnormal"

def main():
    class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

    st.title("Plant Care 🌱")
    #option = st.selectbox("Select an option:", ["Receive Image", "Receive Data"])
    
    # Path to your saved model (replace with your actual path)
    model_path = 'models/100_percent_model_h5.h5'



    # Load the model
    loaded_model = tf.keras.models.load_model(model_path)

    # Get the custom food images file paths
    image_path = "received_image.jpg"
    

    # Create two columns
    col1, col2 = st.columns(2)

    # Button 1 in the first column
    execute = col1.button("Generate Plant Report")

    # Button 2 in the second column
    live_monitoring = col2.button("Start Live Monitoring")


    if execute:
        user_input = 1
        receive_image('1')
        pred_class, pred_prob = predict_class(loaded_model, class_names, image_path)
        # Plot the appropriate information
        display_image()
        st.write(f"pred: {pred_class}, prob: {pred_prob.max():.2f}")
        

        time.sleep(2)
        user_input = 2
        received_data = receive_data('2')
        st.write(received_data)

        image = Image.open(image_path)
        response = get_gemini_response(image, pred_class, pred_prob*100, received_data)
        st.write(response)

    # Initialize chart data
    # Initialize chart data with an empty DataFrame containing the 'Time' column
    chart_data = pd.DataFrame(columns=['Time', 'Temperature', 'Humidity'])
    # Initialize chart if monitoring is started
    


    # Initialize chart if monitoring is started

    if live_monitoring:
         # Button 2 in the second column
        stop_monitoring = col2.button("Stop Live Monitoring")
        monitoring_status = True

        # Create empty placeholders for temperature, humidity, and status
        temp_placeholder = st.empty()
        humidity_placeholder = st.empty()
        status_placeholder = st.empty()
        health_placeholder = st.empty()

        if stop_monitoring:
            monitoring_status = False

        st.subheader("Live Data")
        chart = st.line_chart(chart_data.set_index('Time'))
        temperature =0
        humidity = 0
        # Continuously update chart with new data
        while monitoring_status:
            received_data = receive_data('2')
            
            temperature, humidity = temp_humidity(received_data)
            current_time = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            new_row = {'Time': current_time, 'Temperature': temperature, 'Humidity': humidity}
            chart_data = pd.concat([chart_data, pd.DataFrame([new_row])], ignore_index=True)
            
            # Filter data for the last 20 seconds
            chart_data['Time'] = pd.to_datetime(chart_data['Time'])  # Convert 'Time' column to Timestamp
            chart_data_filtered = chart_data[chart_data['Time'] > pd.Timestamp.now() - pd.Timedelta(seconds=20)]
            
            # Update chart with filtered data
            chart.line_chart(chart_data_filtered.set_index('Time'))
            
            # Update placeholders with current temperature, humidity, and status
            temp_placeholder.text(f"Current Temperature: {temperature}°C")
            humidity_placeholder.text(f"Current Humidity: {humidity}%")
            
            status = check_status(temperature, humidity)
            status_placeholder.text(f"Environment Status: {status}")

            # Fetch and display light intensity every 1 minute
            if int(current_time[-5:-3]) % 1 == 0 and current_time[-2:] == "00":
                receive_image('1')
                pred_class, pred_prob = predict_class(loaded_model, class_names, image_path)
                if "healthy" in pred_class:
                    health_placeholder.text(f"Plant Status: Healthy")
                elif "healthy" not in pred_class and pred_prob.max()>0.5:
                    health_placeholder.text(f"Plant Status: {pred_class}")
                else:
                    health_placeholder.text(f"Plant Status: Healthy")
            
            time.sleep(1)  # Adjust the sleep time as needed for the update frequency

    

if __name__ == "__main__":
    main()
