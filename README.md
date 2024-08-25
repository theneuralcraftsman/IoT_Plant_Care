# IoT-Based Plant Care System 🌱

This project is an IoT-based plant care system that uses a Raspberry Pi and various sensors to monitor and manage the health of plants. The system captures environmental data such as temperature and humidity, takes images of the plants, and provides care recommendations using AI. The system runs a server on the Raspberry Pi that communicates with a client application via WebSocket.

## Features

- **Real-time Monitoring:** Continuously monitor temperature and humidity levels around the plant.
- **AI-Powered Analysis:** Capture images of the plant, analyze them using a deep learning model, and receive care recommendations.
- **Live Data Visualization:** Display real-time data in a user-friendly interface, with live monitoring charts for temperature and humidity.
- **Custom Plant Care Reports:** Generate detailed reports on plant health, leveraging AI to provide suggestions based on live data and images.

## Technologies Used

- **Raspberry Pi**: Acts as the server, capturing sensor data and images.
- **Python**: The primary programming language for server and client-side code.
- **Streamlit**: Used for building the client-side application interface.
- **TensorFlow**: Employed for image classification of plant diseases.
- **Google Generative AI**: Used to generate plant care recommendations based on AI analysis.

## Hardware Requirements

- Raspberry Pi (with GPIO support)
- DHT11 Temperature and Humidity Sensor
- PiCamera
- A server running Python (e.g., Raspberry Pi)

## Installation

### Raspberry Pi (Server-side)

1. **Set up Raspberry Pi:**
   - Install the required libraries:
     ```bash
     sudo apt-get update
     sudo apt-get install python3-pip
     pip3 install Adafruit_DHT picamera
     ```
   - Install the Python packages for socket communication:
     ```bash
     pip3 install socket
     ```

2. **Run the server code on Raspberry Pi:**
   - Place the server script (`server.py`) on the Raspberry Pi.
   - Run the server script:
     ```bash
     python3 server.py
     ```

### Client-side

1. **Set up Python Environment:**
   - Install the required Python packages:
     ```bash
     pip install streamlit tensorflow pillow google-generativeai pandas matplotlib
     ```

2. **Run the client application:**
   - Place the client script (`client.py`) on your local machine.
   - Run the Streamlit application:
     ```bash
     streamlit run client.py
     ```

## Usage

1. **Start the Raspberry Pi server:**
   - Ensure the Raspberry Pi is powered on and connected to your network.
   - Run the server script on the Raspberry Pi.

2. **Use the Client Application:**
   - Open the Streamlit application in your web browser.
   - Choose to start live monitoring or generate a plant care report.
   - The application will display real-time data, captured images, and AI-powered plant care suggestions.

## Example Workflow

1. **Monitor Live Data:**
   - Click on "Start Live Monitoring" in the client app.
   - View real-time temperature and humidity data with live charts.
   - Check the plant's health status based on the captured data.

2. **Generate Plant Report:**
   - Click on "Generate Plant Report".
   - The Raspberry Pi captures an image, and the AI model analyzes it.
   - Receive a detailed plant care report, including temperature, humidity, and image-based analysis.

## Future Enhancements

- **Add more sensors**: Integrate soil moisture sensors, light intensity sensors, etc.
- **Expand AI capabilities**: Train the AI model to recognize more plant diseases.
- **Mobile App**: Develop a mobile app to monitor plants remotely.

