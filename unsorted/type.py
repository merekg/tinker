import keyboard

phrase = "smoke \'em when i want\n lately that\'s a lot"
pressed = False
while(True):
    if not pressed and keyboard.is_pressed('shift'):
        s = phrase.split(' ', 1)
        print(s[0])
        if len(s) > 1:
            phrase = s[1]
        else: break
        pressed = True
    if not keyboard.is_pressed('shift'):
        pressed = False
