import base64
from flask import request, Flask, render_template
from image_processor import process_image, process_ocr_sheet

app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello, World!"

@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    photo_data = None
    result = None
    ocr_img = None
    if request.method == 'POST':
        action = request.form.get('action')
        if 'photo' in request.files:
            photo = request.files['photo']
            if photo.filename:
                img_bytes = photo.read()
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                photo_data = f"data:{photo.mimetype};base64,{img_b64}"
                if action == 'basic':
                    result = process_image(img_bytes)
                elif action == 'ocr':
                    ocr_bytes = process_ocr_sheet(img_bytes)
                    if ocr_bytes:
                        ocr_img = f"data:image/jpeg;base64,{base64.b64encode(ocr_bytes).decode('utf-8')}"
                    else:
                        result = "Could not process OCR sheet."
    return render_template('evaluate.html', photo_data=photo_data, result=result, ocr_img=ocr_img)

if __name__ == '__main__':
    app.run(debug=True)
