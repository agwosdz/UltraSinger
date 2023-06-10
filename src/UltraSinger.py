"""UltraSinger uses AI to automatically create UltraStar song files"""

import copy
import getopt
import os
import sys
from time import sleep

import Levenshtein
import librosa
from tqdm import tqdm

from src.modules import os_helper
from src.modules.Audio.denoise import ffmpeg_reduce_noise
from src.modules.Audio.separation import separate_audio
from src.modules.Audio.vocal_chunks import (
    export_chunks_from_transcribed_data,
    export_chunks_from_ultrastar_data,
)
from src.modules.Audio.silence_processing import remove_silence_from_transcribtion_data
from src.modules.csv_handler import export_transcribed_data_to_csv
from src.modules.Audio.convert_audio import convert_audio_to_mono_wav, convert_wav_to_mp3
from src.modules.Audio.youtube import (
    download_youtube_audio,
    download_youtube_thumbnail,
    download_youtube_video,
    get_youtube_title,
)
from src.modules.DeviceDetection.device_detection import get_available_device
from src.modules.console_colors import (
    ULTRASINGER_HEAD,
    blue_highlighted,
    gold_highlighted,
    light_blue_highlighted,
    red_highlighted,
)
from src.modules.Midi import midi_creator
from src.modules.Midi.midi_creator import (
    convert_frequencies_to_notes,
    create_midi_notes_from_pitched_data,
    most_frequent,
)
from src.modules.Pitcher.pitcher import (
    get_frequency_with_high_confidence,
    get_pitch_with_crepe_file,
)
from src.modules.Pitcher.pitched_data import PitchedData
from src.modules.Speech_Recognition.hyphenation import hyphenation, language_check
from src.modules.Speech_Recognition.Vosk import transcribe_with_vosk
from src.modules.Speech_Recognition.Whisper import transcribe_with_whisper
from src.modules.Ultrastar import ultrastar_score_calculator, ultrastar_writer, ultrastar_converter, ultrastar_parser
from src.modules.Ultrastar.ultrastar_txt import UltrastarTxtValue
from Settings import Settings
from src.modules.Speech_Recognition.TranscribedData import TranscribedData
from src.modules.plot import plot

settings = Settings()


def convert_midi_notes_to_ultrastar_notes(midi_notes: list[str]) -> list[int]:
    """Convert midi notes to ultrastar notes"""
    print(f"{ULTRASINGER_HEAD} Creating Ultrastar notes from midi data")

    ultrastar_note_numbers = []
    for i in enumerate(midi_notes):
        pos = i[0]
        note_number_librosa = librosa.note_to_midi(midi_notes[pos])
        pitch = ultrastar_converter.midi_note_to_ultrastar_note(
            note_number_librosa
        )
        ultrastar_note_numbers.append(pitch)
        # todo: Progress?
        # print(
        #    f"Note: {midi_notes[i]} midi_note: {str(note_number_librosa)} pitch: {str(pitch)}"
        # )
    return ultrastar_note_numbers


def pitch_each_chunk_with_crepe(directory: str) -> list[str]:
    """Pitch each chunk with crepe and return midi notes"""
    print(
        f"{ULTRASINGER_HEAD} Pitching each chunk with {blue_highlighted('crepe')}"
    )

    midi_notes = []
    for filename in sorted(
        [f for f in os.listdir(directory) if f.endswith(".wav")],
        key=lambda x: int(x.split("_")[1]),
    ):
        filepath = os.path.join(directory, filename)
        # todo: stepsize = duration? then when shorter than "it" it should take the duration. Otherwise there a more notes
        pitched_data = get_pitch_with_crepe_file(
            filepath, settings.crepe_step_size, settings.crepe_model_capacity
        )
        conf_f = get_frequency_with_high_confidence(
            pitched_data.frequencies, pitched_data.confidence
        )

        notes = convert_frequencies_to_notes(conf_f)
        note = most_frequent(notes)[0][0]

        midi_notes.append(note)
        # todo: Progress?
        # print(filename + " f: " + str(mean))

    return midi_notes


def add_hyphen_to_data(transcribed_data: list[TranscribedData], hyphen_words: list[list[str]]):
    """Add hyphen to transcribed data return new data list"""
    new_data = []

    for i, data in enumerate(transcribed_data):
        if not hyphen_words[i]:
            new_data.append(data)
        else:
            chunk_duration = data.end - data.start
            chunk_duration = chunk_duration / (len(hyphen_words[i]))

            next_start = data.start
            for j, hyphens in enumerate(hyphen_words[i]):
                dup = copy.copy(data)
                dup.start = next_start
                next_start = data.end - chunk_duration * (
                    len(hyphens) - 1 - j
                )
                dup.end = next_start
                dup.word = hyphen_words[i][j]
                dup.is_hyphen = True
                new_data.append(dup)

    return new_data


def get_bpm_from_data(data, sampling_rate):
    """Get real bpm from audio data"""
    onset_env = librosa.onset.onset_strength(y=data, sr=sampling_rate)
    wav_tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sampling_rate)

    print(
        f"{ULTRASINGER_HEAD} BPM is {blue_highlighted(str(round(wav_tempo[0], 2)))}"
    )
    return wav_tempo[0]


def get_bpm_from_file(wav_file: str):
    """Get real bpm from audio file"""
    data, sampling_rate = librosa.load(wav_file, sr=None)
    return get_bpm_from_data(data, sampling_rate)


def correct_words(recognized_words, word_list_file):
    """Docstring"""
    with open(word_list_file, "r", encoding="utf-8") as file:
        text = file.read()
    word_list = text.split()

    for i, rec_word in enumerate(recognized_words):
        if rec_word.word in word_list:
            continue

        closest_word = min(
            word_list, key=lambda x: Levenshtein.distance(rec_word.word, x)
        )
        print(recognized_words[i].word + " - " + closest_word)
        recognized_words[i].word = closest_word
    return recognized_words


def print_help() -> None:
    """Print help text"""
    help_string = """
    UltraSinger.py [opt] [mode] [transcription] [pitcher] [extra]
    
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
    
    [extra]
    (-k                     Keep audio chunks) # In Progress
    --hyphenation           (default) True|False
    --disable_separation    True|False
    --disable_karaoke       True|False
    
    [pitcher]
    --crepe  (default) tiny|small|medium|large|full
    """
    print(help_string)


def remove_unecessary_punctuations(transcribed_data: list[TranscribedData]) -> None:
    """Remove unecessary punctuations from transcribed data"""
    punctuation = ".,"
    for i, data in enumerate(transcribed_data):
        data.word = data.word.translate(
            {ord(i): None for i in punctuation}
        )


def hyphenate_each_word(language: str, transcribed_data: list[TranscribedData]) -> list[list[str]] | None:
    """Hyphenate each word in the transcribed data."""
    hyphenated_word = []
    lang_region = language_check(language)
    if lang_region is None:
        print(
            f"{ULTRASINGER_HEAD} {red_highlighted('Error in hyphenation for language ')} {blue_highlighted(language)} {red_highlighted(', maybe you want to disable it?')}"
        )
        return None

    sleep(0.1)
    for i, data in tqdm(enumerate(transcribed_data)):
        hyphenated_word.append(
            hyphenation(data.word, lang_region)
        )
    return hyphenated_word


def print_support() -> None:
    """Print support text"""
    print()
    print(
        f"{ULTRASINGER_HEAD} {gold_highlighted('Do you like UltraSinger? And want it to be even better? Then help with your')} {light_blue_highlighted('support')}{gold_highlighted('!')}"
    )
    print(
        f"{ULTRASINGER_HEAD} See project page -> https://github.com/rakuri255/UltraSinger"
    )
    print(
        f"{ULTRASINGER_HEAD} {gold_highlighted('This will help alot to keep this project alive and improved.')}"
    )


def run() -> None:
    """The processing function of this program"""
    is_audio = ".txt" not in settings.input_file_path
    ultrastar_class = None
    real_bpm = None

    if not is_audio:  # Parse Ultrastar txt
        print(
            f"{ULTRASINGER_HEAD} {gold_highlighted('re-pitch mode')}"
        )
        (
            basename_without_ext,
            real_bpm,
            song_output,
            ultrastar_audio_input_path,
            ultrastar_class,
        ) = parse_ultrastar_txt()
    elif settings.input_file_path.startswith("https:"):  # Youtube
        print(
            f"{ULTRASINGER_HEAD} {gold_highlighted('full automatic mode')}"
        )
        (
            basename_without_ext,
            song_output,
            ultrastar_audio_input_path,
        ) = download_from_youtube()
    else:  # Audio File
        print(
            f"{ULTRASINGER_HEAD} {gold_highlighted('full automatic mode')}"
        )
        (
            basename_without_ext,
            song_output,
            ultrastar_audio_input_path,
        ) = infos_from_audio_input_file()

    cache_path = os.path.join(song_output, "cache")
    settings.mono_audio_path = os.path.join(
        cache_path, basename_without_ext + ".wav"
    )
    os_helper.create_folder(cache_path)

    # Separate vocal from audio
    audio_separation_path = separate_vocal_from_audio(
        basename_without_ext, cache_path, ultrastar_audio_input_path
    )

    # Denoise vocal audio
    denoise_vocal_audio(basename_without_ext, cache_path)

    # Audio transcription
    transcribed_data = None
    language = None
    if is_audio:
        language, transcribed_data = transcribe_audio()
        remove_unecessary_punctuations(transcribed_data)
        transcribed_data = remove_silence_from_transcribtion_data(
            settings.mono_audio_path, transcribed_data
        )

        if settings.hyphenation:
            hyphen_words = hyphenate_each_word(language, transcribed_data)
            if hyphen_words is not None:
                transcribed_data = add_hyphen_to_data(
                    transcribed_data, hyphen_words
                )

        # todo: do we need to correct words?
        # lyric = 'input/faber_lyric.txt'
        # --corrected_words = correct_words(vosk_speech, lyric)

    # Create audio chunks
    if settings.create_audio_chunks:
        create_audio_chunks(
            cache_path,
            is_audio,
            transcribed_data,
            ultrastar_audio_input_path,
            ultrastar_class,
        )

    # Pitch the audio
    midi_notes, pitched_data, ultrastar_note_numbers = pitch_audio(
        is_audio, transcribed_data, ultrastar_class
    )

    # Create plot
    if settings.create_plot:
        plot(pitched_data, transcribed_data, midi_notes, song_output)

    # Write Ultrastar txt
    if is_audio:
        real_bpm, ultrastar_file_output = create_ultrastar_txt_from_automation(
            audio_separation_path,
            basename_without_ext,
            song_output,
            transcribed_data,
            ultrastar_audio_input_path,
            ultrastar_note_numbers,
            language,
        )
    else:
        ultrastar_file_output = create_ultrastar_txt_from_ultrastar_data(
            song_output, ultrastar_class, ultrastar_note_numbers
        )

    # Calc Points
    ultrastar_class, simple_score, accurate_score = calculate_score_points(
        is_audio, pitched_data, ultrastar_class, ultrastar_file_output
    )

    # Add calculated score to Ultrastar txt
    ultrastar_writer.add_score_to_ultrastar_txt(
        ultrastar_file_output, simple_score
    )

    # Midi
    if settings.create_midi:
        create_midi_file(is_audio, real_bpm, song_output, ultrastar_class)

    # Print Support
    print_support()


def get_unused_song_output_dir(path: str) -> str:
    """Get an unused song output dir"""
    # check if dir exists and add (i) if it does
    i = 1
    if os_helper.check_if_folder_exists(path):
        path = f"{path} ({i})"
    else:
        return path

    while os_helper.check_if_folder_exists(path):
        path = path.replace(f"({i - 1})", f"({i})")
        i += 1
        if i > 999:
            print(
                f"{ULTRASINGER_HEAD} {red_highlighted('Error: Could not create output folder! (999) is the maximum number of tries.')}"
            )
            sys.exit(1)
    return path


def transcribe_audio() -> (str, list[TranscribedData]):
    """Transcribe audio with AI"""
    if settings.transcriber == "whisper":
        device = "cpu" if settings.force_whisper_cpu else settings.device
        transcribed_data, language = transcribe_with_whisper(
            settings.mono_audio_path, settings.whisper_model, device)
    else:  # vosk
        transcribed_data = transcribe_with_vosk(
            settings.mono_audio_path, settings.vosk_model_path
        )
        # todo: make language selectable
        language = "en"
    return language, transcribed_data


def separate_vocal_from_audio(
        basename_without_ext: str, cache_path: str, ultrastar_audio_input_path: str
) -> str:
    """Separate vocal from audio"""
    audio_separation_path = os.path.join(
        cache_path, "separated", "htdemucs", basename_without_ext
    )
    device = "cpu" if settings.force_separation_cpu else settings.device
    if settings.use_separated_vocal or settings.create_karaoke:
        separate_audio(ultrastar_audio_input_path, cache_path, device)
    if settings.use_separated_vocal:
        vocals_path = os.path.join(audio_separation_path, "vocals.wav")
        convert_audio_to_mono_wav(vocals_path, settings.mono_audio_path)
    else:
        convert_audio_to_mono_wav(
            ultrastar_audio_input_path, settings.mono_audio_path
        )
    return audio_separation_path


def calculate_score_points(
    is_audio: bool, pitched_data: PitchedData, ultrastar_class: UltrastarTxtValue, ultrastar_file_output: str
):
    """Calculate score points"""
    if is_audio:
        ultrastar_class = ultrastar_parser.parse_ultrastar_txt(
            ultrastar_file_output
        )
        (
            simple_score,
            accurate_score,
        ) = ultrastar_score_calculator.calculate_score(
            pitched_data, ultrastar_class
        )
        ultrastar_score_calculator.print_score_calculation(
            simple_score, accurate_score
        )
    else:
        print(
            f"{ULTRASINGER_HEAD} {blue_highlighted('Score of original Ultrastar txt')}"
        )
        (
            simple_score,
            accurate_score,
        ) = ultrastar_score_calculator.calculate_score(
            pitched_data, ultrastar_class
        )
        ultrastar_score_calculator.print_score_calculation(
            simple_score, accurate_score
        )
        print(
            f"{ULTRASINGER_HEAD} {blue_highlighted('Score of re-pitched Ultrastar txt')}"
        )
        ultrastar_class = ultrastar_parser.parse_ultrastar_txt(
            ultrastar_file_output
        )
        (
            simple_score,
            accurate_score,
        ) = ultrastar_score_calculator.calculate_score(
            pitched_data, ultrastar_class
        )
        ultrastar_score_calculator.print_score_calculation(
            simple_score, accurate_score
        )
    return ultrastar_class, simple_score, accurate_score


def create_ultrastar_txt_from_ultrastar_data(
    song_output: str, ultrastar_class: UltrastarTxtValue, ultrastar_note_numbers: list[int]
) -> str:
    """Create Ultrastar txt from Ultrastar data"""
    output_repitched_ultrastar = os.path.join(
        song_output, ultrastar_class.title + ".txt"
    )
    ultrastar_writer.create_repitched_txt_from_ultrastar_data(
        settings.input_file_path,
        ultrastar_note_numbers,
        output_repitched_ultrastar,
    )
    return output_repitched_ultrastar


def create_ultrastar_txt_from_automation(
    audio_separation_path: str,
    basename_without_ext: str,
    song_output: str,
    transcribed_data: list[TranscribedData],
    ultrastar_audio_input_path: str,
    ultrastar_note_numbers: list[int],
    language: str,
):
    """Create Ultrastar txt from automation"""
    ultrastar_header = UltrastarTxtValue()
    ultrastar_header.title = basename_without_ext
    ultrastar_header.artist = basename_without_ext
    ultrastar_header.mp3 = basename_without_ext + ".mp3"
    ultrastar_header.video = basename_without_ext + ".mp4"
    ultrastar_header.language = language
    cover = basename_without_ext + " [CO].jpg"
    ultrastar_header.cover = (
        cover
        if os_helper.check_file_exists(os.path.join(song_output, cover))
        else None
    )

    real_bpm = get_bpm_from_file(ultrastar_audio_input_path)
    ultrastar_file_output = os.path.join(
        song_output, basename_without_ext + ".txt"
    )
    ultrastar_writer.create_ultrastar_txt_from_automation(
        transcribed_data,
        ultrastar_note_numbers,
        ultrastar_file_output,
        ultrastar_header,
        real_bpm,
    )
    if settings.create_karaoke:
        no_vocals_path = os.path.join(audio_separation_path, "no_vocals.wav")
        title = basename_without_ext + " [Karaoke]"
        ultrastar_header.title = title
        ultrastar_header.mp3 = title + ".mp3"
        karaoke_output_path = os.path.join(song_output, title)
        karaoke_audio_output_path = karaoke_output_path + ".mp3"
        convert_wav_to_mp3(no_vocals_path, karaoke_audio_output_path)
        karaoke_txt_output_path = karaoke_output_path + ".txt"
        ultrastar_writer.create_ultrastar_txt_from_automation(
            transcribed_data,
            ultrastar_note_numbers,
            karaoke_txt_output_path,
            ultrastar_header,
            real_bpm,
        )
    return real_bpm, ultrastar_file_output


def infos_from_audio_input_file() -> tuple[str, str, str]:
    """Infos from audio input file"""
    basename = os.path.basename(settings.input_file_path)
    basename_without_ext = os.path.splitext(basename)[0]
    song_output = os.path.join(settings.output_file_path, basename_without_ext)
    song_output = get_unused_song_output_dir(song_output)
    os_helper.create_folder(song_output)
    os_helper.copy(settings.input_file_path, song_output)
    ultrastar_audio_input_path = os.path.join(song_output, basename)
    return basename_without_ext, song_output, ultrastar_audio_input_path


FILENAME_REPLACEMENTS = (('?:"', ""), ("<", "("), (">", ")"), ("/\\|*", "-"))


def sanitize_filename(fname: str) -> str:
    """Sanitize filename"""
    for old, new in FILENAME_REPLACEMENTS:
        for char in old:
            fname = fname.replace(char, new)
    if fname.endswith("."):
        fname = fname.rstrip(" .")  # Windows does not like trailing periods
    return fname


def download_from_youtube() -> tuple[str, str, str]:
    """Download from YouTube"""
    title = get_youtube_title(settings.input_file_path)
    basename_without_ext = sanitize_filename(title)
    basename = basename_without_ext + ".mp3"
    song_output = os.path.join(settings.output_file_path, basename_without_ext)
    song_output = get_unused_song_output_dir(song_output)
    os_helper.create_folder(song_output)
    download_youtube_audio(
        settings.input_file_path, basename_without_ext, song_output
    )
    download_youtube_video(
        settings.input_file_path, basename_without_ext, song_output
    )
    download_youtube_thumbnail(
        settings.input_file_path, basename_without_ext, song_output
    )
    ultrastar_audio_input_path = os.path.join(song_output, basename)
    return basename_without_ext, song_output, ultrastar_audio_input_path


def parse_ultrastar_txt() -> tuple[str, float, str, str, UltrastarTxtValue]:
    """Parse Ultrastar txt"""
    ultrastar_class = ultrastar_parser.parse_ultrastar_txt(
        settings.input_file_path
    )
    real_bpm = ultrastar_converter.ultrastar_bpm_to_real_bpm(
        float(ultrastar_class.bpm.replace(",", "."))
    )
    ultrastar_mp3_name = ultrastar_class.mp3
    basename_without_ext = os.path.splitext(ultrastar_mp3_name)[0]
    dirname = os.path.dirname(settings.input_file_path)
    ultrastar_audio_input_path = os.path.join(dirname, ultrastar_mp3_name)
    song_output = os.path.join(
        settings.output_file_path,
        ultrastar_class.artist + " - " + ultrastar_class.title,
    )
    song_output = get_unused_song_output_dir(song_output)
    os_helper.create_folder(song_output)
    return (
        basename_without_ext,
        real_bpm,
        song_output,
        ultrastar_audio_input_path,
        ultrastar_class,
    )


def create_midi_file(is_audio: bool, real_bpm: float, song_output: str, ultrastar_class: UltrastarTxtValue) -> None:
    """Create midi file"""
    print(
        f"{ULTRASINGER_HEAD} Creating Midi with {blue_highlighted('pretty_midi')}"
    )
    if is_audio:
        voice_instrument = [
            midi_creator.convert_ultrastar_to_midi_instrument(ultrastar_class)
        ]
        midi_output = os.path.join(song_output, ultrastar_class.title + ".mid")
        midi_creator.instruments_to_midi(
            voice_instrument, real_bpm, midi_output
        )
    else:
        voice_instrument = [
            midi_creator.convert_ultrastar_to_midi_instrument(ultrastar_class)
        ]
        midi_output = os.path.join(song_output, ultrastar_class.title + ".mid")
        midi_creator.instruments_to_midi(
            voice_instrument, real_bpm, midi_output
        )


def pitch_audio(is_audio: bool, transcribed_data: list[TranscribedData], ultrastar_class: UltrastarTxtValue) -> tuple[
    list[str], PitchedData, list[int]]:
    """Pitch audio"""
    # todo: chunk pitching as option?
    # midi_notes = pitch_each_chunk_with_crepe(chunk_folder_name)
    pitched_data = get_pitch_with_crepe_file(
        settings.mono_audio_path,
        settings.crepe_step_size,
        settings.crepe_model_capacity,
    )
    if is_audio:
        start_times = []
        end_times = []
        for i, data in enumerate(transcribed_data):
            start_times.append(data.start)
            end_times.append(data.end)
        midi_notes = create_midi_notes_from_pitched_data(
            start_times, end_times, pitched_data
        )

    else:
        midi_notes = create_midi_notes_from_pitched_data(
            ultrastar_class.startTimes, ultrastar_class.endTimes, pitched_data
        )
    ultrastar_note_numbers = convert_midi_notes_to_ultrastar_notes(midi_notes)
    return midi_notes, pitched_data, ultrastar_note_numbers


def create_audio_chunks(
    cache_path: str,
    is_audio: bool,
    transcribed_data: list[TranscribedData],
    ultrastar_audio_input_path: str,
    ultrastar_class: UltrastarTxtValue
) -> None:
    """Create audio chunks"""
    audio_chunks_path = os.path.join(
        cache_path, settings.audio_chunk_folder_name
    )
    os_helper.create_folder(audio_chunks_path)
    if is_audio:  # and csv
        csv_filename = os.path.join(audio_chunks_path, "_chunks.csv")
        export_chunks_from_transcribed_data(
            settings.mono_audio_path, transcribed_data, audio_chunks_path
        )
        export_transcribed_data_to_csv(transcribed_data, csv_filename)
    else:
        export_chunks_from_ultrastar_data(
            ultrastar_audio_input_path, ultrastar_class, audio_chunks_path
        )


def denoise_vocal_audio(basename_without_ext: str, cache_path: str) -> None:
    """Denoise vocal audio"""
    denoised_path = os.path.join(
        cache_path, basename_without_ext + "_denoised.wav"
    )
    ffmpeg_reduce_noise(settings.mono_audio_path, denoised_path)
    settings.mono_audio_path = denoised_path


def main(argv: list[str]) -> None:
    """Main function"""
    init_settings(argv)
    run()
    # todo: cleanup
    sys.exit()


def init_settings(argv: list[str]) -> None:
    """Init settings"""
    long, short = arg_options()
    opts, args = getopt.getopt(argv, short, long)
    if len(opts) == 0:
        print_help()
        sys.exit()
    for opt, arg in opts:
        if opt == "-h":
            print_help()
            sys.exit()
        elif opt in ("-i", "--ifile"):
            settings.input_file_path = arg
        elif opt in ("-o", "--ofile"):
            settings.output_file_path = arg
        elif opt in ("--whisper"):
            settings.transcriber = "whisper"
            settings.whisper_model = arg
        elif opt in ("--vosk"):
            settings.transcriber = "vosk"
            settings.vosk_model_path = arg
        elif opt in ("--crepe"):
            settings.crepe_model_capacity = arg
        elif opt in ("--plot"):
            settings.create_plot = arg
        elif opt in ("--hyphenation"):
            settings.hyphenation = arg
        elif opt in ("--disable_separation"):
            settings.use_separated_vocal = not arg
        elif opt in ("--disable_karaoke"):
            settings.create_karaoke = not arg
        elif opt in ("--create_audio_chunks"):
            settings.create_audio_chunks = arg
        elif opt in ("--force_whisper_cpu"):
            settings.force_whisper_cpu = arg
        elif opt in ("--force_separation_cpu"):
            settings.force_separation_cpu = arg
    if settings.output_file_path == "":
        if settings.input_file_path.startswith("https:"):
            dirname = os.getcwd()
        else:
            dirname = os.path.dirname(settings.input_file_path)
        settings.output_file_path = os.path.join(dirname, "output")
    settings.device = get_available_device()


def arg_options():
    short = "hi:o:amv:"
    long = [
        "ifile=",
        "ofile=",
        "crepe=",
        "vosk=",
        "whisper=",
        "plot=",
        "hyphenation=",
        "disable_separation=",
        "disable_karaoke=",
        "create_audio_chunks=",
        "force_whisper_cpu=",
        "force_separation_cpu="
    ]
    return long, short


if __name__ == "__main__":
    main(sys.argv[1:])