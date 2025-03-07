from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
from ultralytics import YOLO
import cvzone
from deep_translator import GoogleTranslator
from gtts import gTTS
import os

app = Flask(__name__)

model = YOLO("best.pt")
names = model.names
cap = cv2.VideoCapture(0)
count = 0
c = ""
language = "en"

texts = {
    "en": {
        "title": "ISL Live Video Feed",
        "heading": "Live Video Capturing of Indian Sign Language",
        "prompt": "Are you ready for the seamless communication?",
        "class_name": "Class Name"
    },
    "gu": {
        "title": "ISL લાઇવ વિડિઓ ફીડ",
        "heading": "ભારતીય સંકેત ભાષાના લાઇવ વિડિઓ કૅપ્ચરિંગ",
        "prompt": "શું તમે નિરંતર સંચાર માટે તૈયાર છો?",
        "class_name": "વર્ગ નામ"
    },
    "ta": {
        "title": "ISL நேரடி வீடியோ ஒளிபரப்பு",
        "heading": "இந்திய கையெழுத்து மொழியின் நேரடி வீடியோ பதிவு",
        "prompt": "நீங்கள் இடையறாத தொடர்புக்கு தயாரா?",
        "class_name": "வகுப்பு பெயர்"
    },
    "hi": {
        "title": "आईएसएल लाइव वीडियो फीड",
        "heading": "भारतीय सांकेतिक भाषा की लाइव वीडियो कैप्चरिंग",
        "prompt": "क्या आप निर्बाध संचार के लिए तैयार हैं?",
        "class_name": "वर्ग नाम"
    },
    "te": {
        "title": "ISL లైవ్ వీడియో ఫీడ్",
        "heading": "భారతీయ సంకేత భాష యొక్క లైవ్ వీడియో క్యాప్చరింగ్",
        "prompt": "మీరు నిరంతర కమ్యూనికేషన్‌కు సిద్ధంగా ఉన్నారా?",
        "class_name": "తరగతి పేరు"
    },
    "ml": {
        "title": "ISL തത്സമയ വീഡിയോ ഫീഡ്",
        "heading": "ഇന്ത്യൻ സൈൻ ഭാഷയുടെ തത്സമയ വീഡിയോ ക്യാപ്ചറിംഗ്",
        "prompt": "നിരന്തരമായ ആശയവിനിമയത്തിന് നിങ്ങൾ തയ്യാറാണോ?",
        "class_name": "ക്ലാസ് നാമം"
    },
    "bn": {
        "title": "আইএসএল লাইভ ভিডিও ফিড",
        "heading": "ভারতীয় সাইন ভাষার লাইভ ভিডিও ক্যাপচারিং",
        "prompt": "আপনি কি নিরবচ্ছিন্ন যোগাযোগের জন্য প্রস্তুত?",
        "class_name": "শ্রেণীর নাম"
    }
}

def translate_class_name(class_name, lang):
    translated = GoogleTranslator(source='auto', target=lang).translate(class_name)
    return translated

def generate_frames():
    global count, c
    while True:
        ret, frame = cap.read()
        count += 1
        if count % 2 != 0:
            continue
        if not ret:
            break
        
        frame = cv2.resize(frame, (1020, 550))
        results = model.track(frame, persist=True)
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()  
            class_ids = results[0].boxes.cls.int().cpu().tolist()  
            track_ids = results[0].boxes.id.int().cpu().tolist()  
            confidences = results[0].boxes.conf.cpu().tolist()  
            
            for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                global c
                c = names[class_id]
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    global language
    return render_template("index.html", texts=texts[language])

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/frame_count')
def frame_count():
    global c, language
    translated = translate_class_name(c, language)
    return str(translated)

@app.route('/convert_to_speech')
def convert_to_speech():
    global c, language
    translated = translate_class_name(c, language)
    tts = gTTS(text=translated, lang=language)
    tts.save("speech.mp3")
    os.system("start speech.mp3")
    return "Speech conversion done"

@app.route('/set_language', methods=['POST'])
def set_language():
    global language
    language = request.form['language']
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)