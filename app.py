import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
import av
import tensorflow as tf
import numpy as np
import cv2

# Load your machine learning model here
def load_model():
    model = tf.keras.models.load_model('facetracker.h5')  # Replace 'facetracker.h5' with the path to your model file
    return model

class FaceDetector(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.model = load_model()

    def transform(self, frame):
        # Convert frame to RGB
        frame = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)

        # Detect faces
        resized = tf.image.resize(frame, (120, 120))
        yhat = self.model.predict(np.expand_dims(resized / 255, 0))
        sample_coords = yhat[1][0]

        if yhat[0] > 0.5:
            # Controls the main rectangle
            cv2.rectangle(frame,
                          tuple(np.multiply(sample_coords[:2], [frame.shape[1], frame.shape[0]]).astype(int)),
                          tuple(np.multiply(sample_coords[2:], [frame.shape[1], frame.shape[0]]).astype(int)),
                          (255, 0, 0), 2)
            # Controls the label rectangle
            cv2.rectangle(frame,
                          tuple(np.add(np.multiply(sample_coords[:2], [frame.shape[1], frame.shape[0]]).astype(int),
                                      [0, -30])),
                          tuple(np.add(np.multiply(sample_coords[:2], [frame.shape[1], frame.shape[0]]).astype(int),
                                      [80, 0])),
                          (255, 0, 0), -1)

            # Controls the text rendered
            cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [frame.shape[1], frame.shape[0]]).astype(int),
                                                     [0, -5])),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def main():
    st.title('Deep Face Detection Using CNN and OpenCV')
    st.write('This app demonstrates face detection using transfer learning.')

    webrtc_ctx = webrtc_streamer(key="face-detection", video_transformer_factory=FaceDetector)

if __name__ == '__main__':
    main()
