import cv2
import pyttsx3
import time
from PIL import Image
import io
import threading
import numpy as np
import textwrap
from transformers import AutoModelForVision2Seq, AutoProcessor

class ImageCaptioner:
    def __init__(self, model_path='model', processor_path='processor'):
        self.model = AutoModelForVision2Seq.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(processor_path)
    def generate_caption(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption

def caption_camera():
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    captioner = ImageCaptioner(model_path='model', processor_path='processor')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    caption_interval = 2.0
    previous_caption = ""
    current_caption = ""
    caption_ready = True
    last_speak_time = 0
    caption_time_text = ""
    cv2.namedWindow('Camera with Caption', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera with Caption', 800, 600)

    def speak_caption(text):
        nonlocal caption_ready, last_speak_time
        engine.say(text)
        engine.runAndWait()
        last_speak_time = time.time()
        time.sleep(caption_interval)
        caption_ready = True

    def generate_caption_from_image(image_bytes):
        nonlocal current_caption, previous_caption, caption_ready, caption_time_text
        try:
            start_time = time.time()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            caption = captioner.generate_caption(image)
            elapsed_time = time.time() - start_time

            if caption and caption != previous_caption:
                print(f"Caption: {caption}")
                print(f"Caption generated in {elapsed_time:.2f} seconds")
                current_caption = caption
                previous_caption = caption
                caption_time_text = f"Time: {elapsed_time:.2f} seconds"
                caption_ready = False
                threading.Thread(target=speak_caption, args=(caption,), daemon=True).start()

        except Exception as e:
            print(f"Error generating caption: {e}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            h, w, _ = frame.shape
            caption_height = int(h * 0.25)
            total_height = h + caption_height
            canvas = np.ones((total_height, w, 3), dtype=np.uint8) * 255
            canvas[:h, :w] = frame
            if caption_ready:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format='JPEG', quality=50)
                img_bytes = img_byte_arr.getvalue()
                threading.Thread(target=generate_caption_from_image, args=(img_bytes,), daemon=True).start()
                caption_ready = False
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_color = (0, 0, 0)
            thickness = 2
            max_chars_per_line = int(w / 14)
            wrapped_caption = textwrap.wrap(current_caption, width=max_chars_per_line)
            line_height = 25
            start_y = h + 10
            for i, line in enumerate(wrapped_caption):
                text_y = start_y + (i + 1) * line_height
                if text_y < total_height - 5:
                    cv2.putText(canvas, line, (10, text_y), font, font_scale, font_color, thickness)
            if caption_time_text:
                cv2.putText(canvas, caption_time_text, (10, total_height - 10), font, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow('Camera with Caption', canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    caption_camera()