# movie-chatbot
This is a small retrieval-based chatbot that lets you chat with characters from movie scripts like Lord of the rings, Saltburn and Barbie of Swan lake.

## How it works
- loads a movie script where each line is one dialogue.
- builds TF-IDF vectors for all lines.
- turns your query into a TF-IDF vector.
- finds the most similar line using cosine similarity.
- replies with the next line from the script.

## How to run
- requirement: Python 3.8 +
- Clone this repository or download the code
- Open the folder in your terminal or IDE.

Run the program:
- IDE - press "play" - button
- Terminal: python chatbot.py
- Follow the instructions in the terminal

## Credits
- Saltburn and Barbie of Swan Lake scripts: subslikescript.com
- Lord of the Rings (lotr.en): extracted from the corpus OpenSubtitles 2018 (OPUS)
- Developed as part of an assignment in IN2110 at the University of Oslo
