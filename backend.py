from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import time
import traceback
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import numpy as np
from PIL import Image
import io
import google.generativeai as genai
import os
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['GENERATED_FOLDER'] = 'generated'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_FOLDER'], exist_ok=True)
 
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.enable_attention_slicing()
pipe.enable_model_cpu_offload()

 
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

def generate_room_redesign(input_image):
    prompt = """
    [Professional interior design shot], [modern living room], 
    [MINIMAL CHANGES: strictly preserve wall structure, floor material, window shape, window position], 
    [ONLY UPDATE: furniture arrangement, decor items, lighting fixtures],
    [Dont change the windows of the room into anything keep it as it is]
    [Style: contemporary], [Color palette: neutral tones], 
    [Lighting: soft natural light through windows]
    [Dont change the floor look keep it original]
    """
    
    negative_prompt = """
    (poor quality:1.3), (distorted:1.2), (cluttered:1.2), 
    (structural changes:1.4), (altered windows:1.3), 
    (changed flooring:1.3), (unrealistic proportions:1.2)
    (too many things)
    """
    
    result = pipe(
        prompt=prompt,
        image=input_image,
        negative_prompt=negative_prompt,
        num_inference_steps=35,
        guidance_scale=8.0,
        controlnet_conditioning_scale=0.8,
        generator=torch.Generator(device="cuda").manual_seed(42)
    )
    return result.images[0]

def generate_description(original_path, redesigned_path):
    try:
        original_img = Image.open(original_path)
        redesigned_img = Image.open(redesigned_path)

        prompt = """
        Analyze these two interior design images (original and redesigned) and provide a 2-3 sentence comparison.
        Focus on:
        - What elements were preserved
        - What design changes were made
        - The overall style difference
        Be concise and professional.
        """

        response = gemini_model.generate_content([prompt, original_img, redesigned_img])
        return response.text

    except Exception as e:
        print(f"[Gemini Error] {str(e)}")
        return "Redesign completed successfully, but description generation failed."

def overlay_object_on_image(base_img: Image.Image, obj_img: Image.Image, position: tuple):
    """Overlay obj_img on base_img at position (x, y)"""
    base = base_img.convert("RGBA")
    obj = obj_img.convert("RGBA")
    x, y = position
     
    x = max(0, min(x, base.width - obj.width))
    y = max(0, min(y, base.height - obj.height))
    base.paste(obj, (x, y), mask=obj)
    return base.convert("RGB")

@app.route('/generate', methods=['POST'])
def generate_design():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'error': 'No image file in request'}), 400
    
    try:
        file = request.files['image']
        if not file or file.filename == '':
            return jsonify({'status': 'error', 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'status': 'error', 'error': 'Only JPG/PNG files allowed'}), 400
        
        filename = secure_filename(f"upload_{int(time.time())}_{file.filename}")
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        redesigned_path = os.path.join(app.config['GENERATED_FOLDER'], f"redesign_{filename}")
        
        file.save(original_path)
        
        input_image = load_image(original_path)
        redesigned_image = generate_room_redesign(input_image)
        redesigned_image.save(redesigned_path)
        
        description = generate_description(original_path, redesigned_path)
        
        return jsonify({
            'status': 'success',
            'original': filename,
            'redesigned': f"redesign_{filename}",
            'description': description
        })
        
    except Exception as e:
        if 'original_path' in locals() and os.path.exists(original_path):
            os.remove(original_path)
        if 'redesigned_path' in locals() and os.path.exists(redesigned_path):
            os.remove(redesigned_path)
            
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/add-object', methods=['POST'])
def add_object():
    try:
         
        room_filename = request.files['room'].filename
        if not room_filename:
            return jsonify({'status': 'error', 'error': 'No room filename provided'}), 400
        
        room_path = os.path.join(app.config['GENERATED_FOLDER'], room_filename)
        if not os.path.exists(room_path):
            return jsonify({'status': 'error', 'error': 'Room image not found on server'}), 404
        
     
        room_img = Image.open(room_path).convert("RGB")
 
        if 'object' not in request.files:
            return jsonify({'status': 'error', 'error': 'No object image uploaded'}), 400
        obj_file = request.files['object']
        obj_img = Image.open(io.BytesIO(obj_file.read())).convert("RGBA")
 
        try:
            x = int(request.form.get('x'))
            y = int(request.form.get('y'))
        except Exception:
            return jsonify({'status': 'error', 'error': 'Invalid or missing coordinates'}), 400
 
        composite_img = overlay_object_on_image(room_img, obj_img, (x, y))
 
        prompt = "A stylish sofa placed naturally in a modern living room. Keep everything else unchanged."
        negative_prompt = "blur, distorted, low quality, unrealistic"
 
        result = pipe(
            prompt=prompt,
            image=composite_img,
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            generator=torch.Generator(device="cuda").manual_seed(int(time.time()))
        )
        generated_img = result.images[0]

        # Save generated image
        out_filename = f"addobj_{uuid.uuid4().hex}.png"
        out_path = os.path.join(app.config['GENERATED_FOLDER'], out_filename)
        generated_img.save(out_path)

        return jsonify({
            'status': 'success',
            'generated_image': out_filename
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/generated/<filename>')
def generated_file(filename):
    return send_from_directory(app.config['GENERATED_FOLDER'], filename)

@app.route('/')
def serve_index():
    return send_from_directory('static', 'Interiordesigner.html')

@app.route('/design.html')
def serve_design():
    return send_from_directory('static', 'design.html')

@app.route('/edit.html')
def serve_edit():
    return send_from_directory('static', 'edit.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
