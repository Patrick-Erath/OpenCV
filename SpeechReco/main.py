import speech_recognition as sr

r = sr.Recognizer()
with sr.Microphone() as source:
    print('Say Something')
    audio = r.listen(source)

x = True

try:
    print(r.recognize_sphinx(audio))
except Exception as e:
    print('error '+str(e))