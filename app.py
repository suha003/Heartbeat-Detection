from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.signal import find_peaks,butter, lfilter

app=Flask(__name__)

camera = None
bpm = 0
def capture_by_frames(capture): 
    global camera,bpm
    trace = []
    while capture:
        if camera is not None:
            success, frame = camera.read()  # read the camera frame
            if success and frame is not None:
                detector=cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
                faces=detector.detectMultiScale(frame,1.2,6)
                #Draw the rectangle around each face
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    cv2.putText(frame, f'BPM: {bpm}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                forehead=detector.detectMultiScale(frame,1.2,6)
                #Draw the rectangle for forehead
                if len(faces) > 0:
                    x,y,w,h = faces[0]
                for (x1, y1, w1, h1) in forehead:
                    y1 = y + 10
                    h1 = h//5
                    w1 = w//3
                    x1  = x + (w//2) - (w1//2)
                    cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 3)
                    forehead_frame = frame[y1:y1+h1, x1:x1+w1]
                    #cv2.imwrite('C:/Users/chinm/OneDrive/Pictures/Saved Pictures/test.jpg', forehead_frame)
                    forehead_trace = np.mean(forehead_frame,axis=1)
                    mean = np.mean(forehead_trace)
                    std = np.std(forehead_trace)
                    normalized_trace = (forehead_trace - mean) / std
                    if not np.log2(len(normalized_trace)).is_integer():
                        padded_length = int(2**(np.ceil(np.log2(len(normalized_trace)))))
                        normalized_trace = np.pad(normalized_trace, (0, padded_length-len(normalized_trace)), mode='constant')
                        trace.append(normalized_trace)
                        #print(normalized_trace)
                    #print(forehead_trace)
                    #print(trace)

                    fft_signal = fft(normalized_trace)
                    freqs = fftfreq(len(normalized_trace))
                    #print(freqs)
                    fft_signal_filtered = fft_signal.copy()
                    # Define the filter parameters
                    low_cutoff = 0.75
                    high_cutoff = 5
                    filter_order = 1

                    # Define the sampling rate
                    fs = 25

                    # Create a filter
                    b, a = butter(filter_order, [low_cutoff / (fs / 2), high_cutoff / (fs / 2)], btype='band')

                    # Filter the signal
                    filtered_forehead_trace = lfilter(b, a, forehead_trace)

                    # Find peaks in the filtered signal
                    peaks, _ = find_peaks(np.ravel(filtered_forehead_trace), distance=100)

                    # Calculate the BPM
                    if len(peaks) > 0:
                        bpm = int(60 / np.mean(np.diff(peaks)) * fs * len(forehead_trace) / (high_cutoff - low_cutoff)-36)
                        print("BPM:", bpm)
                    else:
                        print("No peaks detected.")


                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg',frame)
                    if ret:
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')





@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start',methods=['POST'])
def start():
    global camera
    camera = cv2.VideoCapture(0)
    return render_template('index.html')

@app.route('/stop',methods=['POST'])
def stop():
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
    camera = None
    return render_template('stop.html')

@app.route('/video_capture')
def video_capture():
    global camera
    if camera is None or not camera.isOpened():
        return Response('Camera not available')
    return Response(capture_by_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/bpm')
def get_bpm():
    global bpm
    return jsonify({'bpm': bpm})

if __name__=='__main__':
    app.run(debug=True,use_reloader=False, port=8000)