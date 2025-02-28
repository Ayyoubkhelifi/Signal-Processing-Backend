from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)  # Allow requests from Flutter

# Define basic functions
def rect(t):
    return np.where(np.abs(t) <= 0.5, 1, 0)

def tri(t):
    return np.where(np.abs(t) <= 1, 1 - np.abs(t), 0)

def u(t):
    return np.where(t >= 0, 1, 0)

def delta(t):
    return np.where(t == 0, 1, 0)  # Approximate delta function

# Define signals
def compute_signal(signal_id, t):
    if signal_id == "x1":
        return 2 * np.real(rect(t - 1))
    elif signal_id == "x2":
        return np.sin(np.pi * np.real(rect(t / 2)))
    elif signal_id == "x3":
        return tri(2 * t)
    elif signal_id == "x4":
        return u(t - 2)
    elif signal_id == "x5":
        return delta(t + 1) - delta(t - 2) + delta(t - 1)
    elif signal_id == "x6":
        return np.real(rect(t + 2)) - np.real(rect(t - 1))
    elif signal_id == "x7":
        return np.exp(-t) * u(t - 2)
    elif signal_id == "x8":
        return np.sin(4 * np.pi * t)  # sin(4πt), frequency = 2Hz
    else:
        return np.zeros_like(t)  # Default: return zeros if signal_id is invalid

@app.route('/get_signal', methods=['POST'])
def get_signal():
    data = request.json
    signal_id = data.get("signal_id", "x1")
    t = np.linspace(-5, 5, 500)  # Time range
    
    y = compute_signal(signal_id, t)
    
    # Plot the signal
    plt.figure(figsize=(6, 4))
    plt.plot(t, y, label=signal_id)
    plt.xlabel("Time (t)")
    plt.ylabel("Amplitude")
    plt.title(f"Signal {signal_id}")
    plt.legend()
    plt.grid()

    # Save plot as a base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded_image = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return jsonify({"image": encoded_image})

if __name__ == '__main__':
    app.run(debug=True)
