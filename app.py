import os
import cv2
import torch
import numpy as np
import base64
import json
import threading
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from torchvision import transforms
from model import CSRNet

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ─── Model ───────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Kullanilan cihaz: {device}")

model = CSRNet(load_weights=True)
checkpoint = torch.load('weights.pth', map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()
print("Model hazir!")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ─── Global stop flag ────────────────────────────────────────
stop_flag = threading.Event()

# ─── Kare İşle ───────────────────────────────────────────────

def process_frame(frame):
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w  = rgb.shape[:2]
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    rgb_r = cv2.resize(rgb, (new_w, new_h))

    img_tensor = transform(rgb_r).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)

    count   = output.sum().item()
    density = output.squeeze().cpu().numpy()
    density = cv2.resize(density, (w, h))
    density = np.maximum(density, 0)

    if density.max() > 0:
        density_norm = (density / density.max() * 255).astype(np.uint8)
    else:
        density_norm = np.zeros((h, w), dtype=np.uint8)

    # Sadece saf ısı haritası — video karesiyle harmanlanmıyor
    heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)

    return heatmap, int(count)

def frame_to_base64(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode('utf-8')

# ─── Routes ──────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        return jsonify({'error': 'Video bulunamadi'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Dosya secilmedi'}), 400
    filename   = file.filename
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)
    return jsonify({'success': True, 'filename': filename})

@app.route('/stop', methods=['POST'])
def stop():
    stop_flag.set()
    return jsonify({'stopped': True})

@app.route('/stream/<filename>')
def stream(filename):
    input_path  = os.path.join(UPLOAD_FOLDER, filename)
    output_name = 'output_' + filename
    output_path = os.path.join(OUTPUT_FOLDER, output_name)

    # Her yeni stream başında flag'i sıfırla
    stop_flag.clear()

    def generate():
        cap    = cv2.VideoCapture(input_path)
        fps    = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        counts    = []
        frame_idx = 0

        while cap.isOpened():
            # Durdurma kontrolü
            if stop_flag.is_set():
                payload = json.dumps({'type': 'stopped', 'counts': counts})
                yield f"data: {payload}\n\n"
                break

            ret, frame = cap.read()
            if not ret:
                break

            processed, count = process_frame(frame)
            counts.append(count)
            out.write(processed)
            frame_idx += 1

            if frame_idx % 3 == 0 or frame_idx == 1:
                b64     = frame_to_base64(processed)
                payload = json.dumps({
                    'type'      : 'frame',
                    'frame'     : b64,
                    'count'     : count,
                    'total_count': sum(counts),
                    'frame_idx' : frame_idx,
                    'total'     : total,
                    'progress'  : round(frame_idx / total * 100, 1)
                })
                yield f"data: {payload}\n\n"

        cap.release()
        out.release()

        if not stop_flag.is_set():
            final = json.dumps({
                'type'        : 'done',
                'output_video': f'/static/outputs/{output_name}',
                'counts'      : counts,
                'avg_count'   : round(float(np.mean(counts)), 1) if counts else 0,
                'max_count'   : int(max(counts)) if counts else 0,
                'min_count'   : int(min(counts)) if counts else 0,
                'total_frames': total
            })
            yield f"data: {final}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control'    : 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

if __name__ == '__main__':
    app.run(debug=False, port=5000, threaded=True)