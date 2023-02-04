# UltraSinger 

<a href="https://www.buymeacoffee.com/rakuri255" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

_This project is still under development!_

_The output files from the full automatic are currently not really usable!
But it is usable for re-pitching ultrastar files._

UltraSinger is a tool to automatically create UltraStar.txt, midi and notes from music. 
It also can re-pitch current UltraStar files.

Multiple AI models are used to extract text from the voice and to determine the pitch. 

## Requirement

You need FFmpeg installed.

## How to use

_Not all options working now!_
```commandline
    UltraSinger.py [opt] [mode] [transcription] [rec model] [pitcher] [extra]
    
    [opt]
    -h      This help text.
    -i      Ultrastar.txt
            audio like .mp3, .wav, youtube link
    -o      Output folder
    
    [mode]
    ## INPUT is audio ##
    default  Creates all
            
    (-u      Create ultrastar txt file) # In Progress
    (-m      Create midi file) # In Progress
    (-s      Create sheet file) # In Progress
    
    ## INPUT is ultrastar.txt ##
    default  Creates all

    (-r      repitch Ultrastar.txt (input has to be audio)) # In Progress
    (-p      Check pitch of Ultrastar.txt input) # In Progress
    (-m      Create midi file) # In Progress

    [transcription]
    --whisper   (default) tiny|base|small|medium|large
    --vosk      Needs model
    
    [rec model]
    (-k      Keep audio chunks) # In Progress
      
    [extra]
    -k      Keep audio chunks
    
    [pitcher]
    --crepe  (default) tiny|small|medium|large|full
    '''
```

### Input

#### Audio (full automatic)

##### Local file

```commandline
-i "input/music.mp3"
```

##### Youtube

```commandline
-i https://www.youtube.com/watch?v=BaW_jenozKc
```

#### UltraStar (re-pitch)

This re-pitch the audio and creates a new txt file.

```commandline
-i "input/ultrastar.txt"
```

### Transcriber

For transcription, `whisper` is used by default. It is more accurate than the other even with the `tiny` model.
And it finds the language automatically.
But anyway, it depends! Try the other one if `Whisper` does not suit you.
Also keep in mind that while a larger model is more accurate, it also takes longer to transcribe.

#### Whisper

For the first test run, use the `tiny`, to be accurate use the `large` model

```commandline
-i XYZ --whisper large
```

#### Vosk

If you want to use `Vosk` than you need the model. It is not included. You can download it here [Link](https://alphacephei.com/vosk/models).
Make sure you take the right language. 
For the first test run, use the `small model`, to be accurate use the `gigaspeech` model

```commandline
-i "input/music.mp3" -v "models\vosk-model-en-us-0.42-gigaspeech"
```

### Pitcher

Pitching is done with the `crepe` model. 
Also consider that a bigger model is more accurate, but also takes longer to pitch.
For just testing you should use `tiny`, which is currently default.
If you want solid accurate, then use the `full` model.

```commandline
-i XYZ --crepe full
```
