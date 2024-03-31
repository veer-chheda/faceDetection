import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Load your machine learning model here
def load_model():
    model = tf.keras.models.load_model('facetracker.h5')  # Replace 'facetracker.h5' with the path to your model file
    return model

def main():
    st.title('Deep Face Detection Using CNN and OpenCV')
    st.write('This app demonstrates face detection using transfer learning.')

    # Load the model
    model = load_model()

    # Button to start face detection
    if st.button('Start Face Detection'):
        # Initialize the video capture object
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        stop_button_pressed = st.button("Stop")
        # Start video streaming
        while cap.isOpened() and not stop_button_pressed:
            # Read frame from video capture
            ret, frame = cap.read()
            
            if not ret:
                st.write("Video Capture Ended")
                break

            # Detect faces
            frame = frame[10:690, 50:1230, :]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = tf.image.resize(rgb, (120, 120))
            
            yhat = model.predict(np.expand_dims(resized/255,0))
            sample_coords = yhat[1][0]
            
            if yhat[0] > 0.5: 
                # Controls the main rectangle
                cv2.rectangle(frame, 
                            tuple(np.multiply(sample_coords[:2], [1180, 680]).astype(int)),
                            tuple(np.multiply(sample_coords[2:], [1180, 680]).astype(int)), 
                                    (255,0,0), 2)
                # Controls the label rectangle
                cv2.rectangle(frame, 
                            tuple(np.add(np.multiply(sample_coords[:2], [1180, 680]).astype(int), 
                                            [0,-30])),
                            tuple(np.add(np.multiply(sample_coords[:2], [1180, 680]).astype(int),
                                            [80,0])), 
                                    (255,0,0), -1)
                
                # Controls the text rendered
                cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [1180, 680]).astype(int),
                                                    [0,-5])),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            
            # Convert the frame to RGB color space
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame using OpenCV (not Streamlit)
            frame_placeholder.image(frame_rgb, channels="RGB")

            # Check for user input to exit
            if stop_button_pressed:
                break

        # Release the video capture object
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
