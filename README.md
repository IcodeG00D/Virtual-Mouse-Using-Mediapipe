
# Virtual Mouse Using Hand Gestures

## Overview

This project implements a **virtual mouse** using hand gestures via your webcam. It uses **MediaPipe** for real-time hand tracking and **PyAutoGUI** to control the mouse, scroll, and click — all with your fingers. Ideal for touchless control or fun automation.

---

## 📌 Features

* **Real-time hand tracking** using MediaPipe
* Control mouse **cursor with your index finger**
* **Scroll up/down** with two fingers or fist gestures
* Perform **mouse click** with all fingers extended
* Exit using the **“Hang Loose” gesture** 🤙 (thumb and pinky extended)
* Uses **gesture-based UI** — no physical mouse required

---

## 🎯 Technologies Used

| Library        | Purpose                                        |
| -------------- | ---------------------------------------------- |
| `cv2` (OpenCV) | Captures video from webcam & visualizes output |
| `mediapipe`    | Detects and tracks hand landmarks              |
| `pyautogui`    | Controls the mouse and keyboard                |
| `time`         | Manages delays and debounce functionality      |

---

## 🤖 Why These Libraries?

* **MediaPipe**: Lightweight and efficient ML framework by Google for hand/pose tracking.
* **PyAutoGUI**: Easy-to-use Python library to control mouse and keyboard events.
* **OpenCV**: For accessing camera feed and overlaying visual cues (like finger landmarks).
* **Time**: Prevents rapid/unintended repeated actions (debounce logic).

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/virtual-mouse-gestures.git
cd virtual-mouse-gestures
```

### 2. Set up a virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> `requirements.txt` should contain:

```txt
opencv-python
mediapipe
pyautogui
```

### 4. Run the program

```bash
python your_script_name.py
```

---

## ✋ Gesture Mappings

| Gesture                       | Action       |
| ----------------------------- | ------------ |
| ☝️ Index finger up            | Move mouse   |
| ✌️ Index + middle finger      | Scroll up    |
| ✊ Fist                        | Scroll down  |
| 🖐️ All fingers extended      | Click        |
| 🤙 Hang Loose (thumb + pinky) | Exit program |

---

## 📌 Notes

* Works best in well-lit environments.
* Webcam resolution and frame rate affect performance.
* Smoothing and debounce can be fine-tuned via constants:

  * `SMOOTHING_FACTOR`
  * `DEBOUNCE_TIME`

---

## 🧪 Future Improvements

* Add gesture for **right-click or drag**.
* Include **volume control** or **media gestures**.
* Show **on-screen cursor overlay**.
* Use **Kalman filter** or **AI-based stabilization** for smoother movement.

