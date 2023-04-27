# Ascii Art Generator

## Intro
Use: python3 AsciiArt.py `path/to/image.png`
Outputs: a string of ascii art to the console.

This art generator will take in an image, try to bin it down to a reasonable if possible, then convert the image into a grayscale image, then convert the grayscale into a set of characters that will mimic intensity. For instance, '.' is a very low intensity character whereas '@' is a very high intensity character.
