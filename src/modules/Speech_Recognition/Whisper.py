"""Whisper Speech Recognition Module"""
import inspect
import textwrap
import torch
import whisperx
from enum import Enum
from torch.cuda import OutOfMemoryError

from modules.Speech_Recognition.TranscriptionResult import TranscriptionResult
from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted
from modules.Speech_Recognition.TranscribedData import TranscribedData, from_whisper

#Addition for numbers to words
import re
import ast
import num2words
import re
from num2words.lang_EN import Num2Word_EN

#Addition for numbers to words_v2
re_split_preserve_space = re.compile(  
    r'''  
    # Currency amounts with commas (match first)  
    [£$€][0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?  
    |  
    # Currency amounts without commas  
    [£$€][0-9]+(?:\.[0-9]+)?  
    |  
    # Numbers with commas  
    [0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]+)?  
    |  
    # Numbers without commas  
    [0-9]+(?:\.[0-9]+)?  
    |  
    # Words  
    [A-Za-z]+  
    |  
    # Spaces (preserve)  
    \s+  
    |  
    # Punctuation  
    [^\w\s]  
    ''',   
    re.VERBOSE  
)  


MEMORY_ERROR_MESSAGE = f"{ULTRASINGER_HEAD} {blue_highlighted('whisper')} ran out of GPU memory; reduce --whisper_batch_size or force usage of cpu with --force_cpu"

class WhisperModel(Enum):
    """Whisper model"""
    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V1 = "large-v1"
    LARGE_V2 = "large-v2"
    LARGE_V3 = "large-v3"

#Addition for num2words
def to_currency(self, val, currency='USD', cents=True, separator=' -',
        adjective=False):
    result = super(Num2Word_EN, self).to_currency(
        val, currency=currency, cents=cents, separator=separator,
        adjective=adjective)
    # Handle exception, in german is "ein Euro" and not "eins Euro"
    return result.replace("zero dollars - ", "")


def is_year(number, min_year=1000, max_year=None):  
    """  
    Evaluate if a number is likely a year.  
      
    :param number: The number to evaluate.  
    :param min_year: The minimum value for a year. Default is 1000.  
    :param max_year: The maximum value for a year. Default is the current year.  
    :return: True if the number is within the range of years, otherwise False.  
    """  
    if max_year is None:  
        from datetime import datetime  
        max_year = datetime.now().year  
      
    return isinstance(number, int) and min_year <= number <= max_year  
    
#Addition for numbers to words (Using previous code from louispan in PR#135)
def number_to_words(line, language='en'):  
    out_tokens = []  
    in_tokens = [match.group(0) for match in re_split_preserve_space.finditer(line)]  
    for token in in_tokens: 
        try:  
            if token[0] in '$€£¥₹₽₩₪฿₫₴₦₱₲₵₸₺₡₭₮₦₣₳':  
                # Extract the numeric part and convert  
                currency_symbol = token[0]  
                currency_number = token[1:].replace(',', '')  # Remove commas for conversion
                num_in_words = num2words.num2words(float(currency_number), lang=language, to='currency')  
                out_tokens.append(f"{currency_symbol}{num_in_words}")  
            else:  
                # Try to evaluate as a number  
                num = ast.literal_eval(token)
                if is_year(num):
                    out_tokens.append(num2words.num2words(num, lang=language, to='year'))  
                else:  
                    out_tokens.append(num2words.num2words(num, lang=language))  
        except Exception:  
            # If not a number, append the token as is  
            out_tokens.append(token)  
    return ''.join(out_tokens)  

def replace_code_lines(source, start_token, end_token,
                       replacement, escape_tokens=True):
    """Replace the source code between `start_token` and `end_token`
    in `source` with `replacement`. The `start_token` portion is included
    in the replaced code. If `escape_tokens` is True (default),
    escape the tokens to avoid them being treated as a regular expression."""

    if escape_tokens:
        start_token = re.escape(start_token)
        end_token = re.escape(end_token)

    def replace_with_indent(match):
        indent = match.group(1)
        return textwrap.indent(replacement, indent)

    return re.sub(r"^(\s+)({}[\s\S]+?)(?=^\1{})".format(start_token, end_token),
                  replace_with_indent, source, flags=re.MULTILINE)

def transcribe_with_whisper(
    audio_path: str,
    model: WhisperModel,
    device="cpu",
    alignment_model: str = None,
    batch_size: int = 16,
    compute_type: str = None,
    language: str = None,
    keep_numbers: bool = False,
) -> TranscriptionResult:
    """Transcribe with whisper"""
    # Info: Monkey Patch FasterWhisperPipeline.detect_language to include error handling for low confidence
    src = textwrap.dedent(inspect.getsource(whisperx.asr.FasterWhisperPipeline.detect_language))
    # Replace the relevant part of the method
    start_token = "if audio.shape[0] < N_SAMPLES:"
    end_token = "return language"
    replacement = """\
    #Added imports
    from modules.console_colors import ULTRASINGER_HEAD, blue_highlighted, red_highlighted
    from Settings import Settings
    from inputimeout import inputimeout, TimeoutOccurred
    #End Import addition
    if audio.shape[0] < N_SAMPLES:
        print("Warning: audio is shorter than 30s, language detection may be inaccurate.")
    model_n_mels = self.model.feat_kwargs.get("feature_size")
    segment = log_mel_spectrogram(audio[: N_SAMPLES],
                                    n_mels=model_n_mels if model_n_mels is not None else 80,
                                    padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
    encoder_output = self.model.encode(segment)
    results = self.model.model.detect_language(encoder_output)
    language_token, language_probability = results[0][0]
    language = language_token[2:-2]
    print(f"Detected language: {language} ({language_probability:.2f}) in first 30s of audio...")
    #Added handling for low detection probability
    if language_probability < Settings.CONFIDENCE_THRESHOLD:
        print(f"{ULTRASINGER_HEAD} {red_highlighted('Warning:')} Language detection probability for detected language {language} is below {Settings.CONFIDENCE_THRESHOLD}, results may be inaccurate.")
        print(f"{ULTRASINGER_HEAD} Override the language below or re-run with parameter {blue_highlighted('--language xx')} to specify the song language...")    
        try:  
            response = inputimeout(  
                prompt=f"{ULTRASINGER_HEAD} Do you want to continue with {language} (default) or override with another language (y)? (y/n): ",  
                timeout=Settings.CONFIDENCE_PROMPT_TIMEOUT  
            ).strip().lower()  
        except TimeoutOccurred:
            import locale
            print(f"{ULTRASINGER_HEAD} No user input received in {Settings.CONFIDENCE_PROMPT_TIMEOUT} seconds. Attempting automatic override with system locale.")
            print(f"{ULTRASINGER_HEAD} Trying to get language from default locale...")  
            current_locale = locale.getlocale()
            if current_locale[0]:  
                language_code = current_locale[0][:2].strip().lower()
                print(f"{ULTRASINGER_HEAD} Found language code: {language_code} in locale. Setting language to {blue_highlighted(language_code)}...")
                language = language_code   
            else:  
                print(f"{ULTRASINGER_HEAD} No locale is set.")  
            response = 'n'
        language_response = response == 'y'  
        if language_response:
            language = input(f"{ULTRASINGER_HEAD} Please enter the language code for the language you want to use (e.g. 'en', 'de', 'es', etc.): ").strip().lower()
    #End addition
    """
    new_src = replace_code_lines(src, start_token, end_token, replacement)
    # Compile it and execute it in the target module's namespace
    exec(compile(new_src, "<string>", "exec"), whisperx.asr.__dict__)
    whisperx.asr.FasterWhisperPipeline.detect_language = whisperx.asr.detect_language
    #End Monkey Patch

    # Info: Regardless of the audio sampling rate used in the original audio file, whisper resample the audio signal to 16kHz (via ffmpeg). So the standard input from (44.1 or 48 kHz) should work.

    print(
        f"{ULTRASINGER_HEAD} Loading {blue_highlighted('whisper')} with model {blue_highlighted(model.value)} and {red_highlighted(device)} as worker"
    )
    if alignment_model is not None:
        print(f"{ULTRASINGER_HEAD} using alignment model {blue_highlighted(alignment_model)}")

    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    try:
        torch.cuda.empty_cache()
        loaded_whisper_model = whisperx.load_model(
            model.value, language=language, device=device, compute_type=compute_type
        )

        audio = whisperx.load_audio(audio_path)

        print(f"{ULTRASINGER_HEAD} Transcribing {audio_path}")

        result = loaded_whisper_model.transcribe(
            audio, batch_size=batch_size, language=language
        )

        detected_language = result["language"]
        if language is None:
            language = detected_language

        # load alignment model and metadata
        try:
            model_a, metadata = whisperx.load_align_model(
                language_code=language, device=device, model_name=alignment_model
            )
        except ValueError as ve:
            print(
                f"{red_highlighted(f'{ve}')}"
                f"\n"
                f"{ULTRASINGER_HEAD} {red_highlighted('Error:')} Unknown language. "
                f"Try add it with --align_model [huggingface]."
            )
            raise ve

        #Addition for numbers to words (Using previous code from louispan in PR#135)
        if keep_numbers == False: 
            Num2Word_EN.to_currency = to_currency  
            for obj in result["segments"]:
                print(f"laanguage: {language}")
                print( obj["text"])
                obj["text"] = number_to_words(obj["text"],language)
                print(obj["text"])
        # align whisper output
        result_aligned = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        transcribed_data = convert_to_transcribed_data(result_aligned)

        return TranscriptionResult(transcribed_data, detected_language)
    except ValueError as value_error:
        if (
            "Requested float16 compute type, but the target device or backend do not support efficient float16 computation."
            in str(value_error.args[0])
        ):
            print(value_error)
            print(
                f"{ULTRASINGER_HEAD} Your GPU does not support efficient float16 computation; run UltraSinger with '--whisper_compute_type int8'"
            )

        raise value_error
    except OutOfMemoryError as oom_exception:
        print(oom_exception)
        print(MEMORY_ERROR_MESSAGE)
        raise oom_exception
    except Exception as exception:
        if "CUDA failed with error out of memory" in str(exception.args[0]):
            print(exception)
            print(MEMORY_ERROR_MESSAGE)
        raise exception


def convert_to_transcribed_data(result_aligned):
    transcribed_data = []
    for segment in result_aligned["segments"]:
        for obj in segment["words"]:
            vtd = from_whisper(obj)  # create custom Word object
            vtd.word = vtd.word + " "  # add space to end of word
            if len(obj) < 4:
                #Addition for numbers to words (Using previous code from louispan in PR#135)
                if len(transcribed_data) == 0: # if the first word doesn't have any timing data
                    vtd.start = 0.0
                    vtd.end = 0.1
                    msg = f'Error: There is no timestamp for word: "{obj["word"]}". ' \
                        f'Fixing it by placing it at beginning. At start: {vtd.start} end: {vtd.end}. Fix it manually!'
                else:
                    previous = transcribed_data[-1] if len(transcribed_data) != 0 else TranscribedData()
                    vtd.start = previous.end + 0.1
                    vtd.end = previous.end + 0.2
                    msg = f'Error: There is no timestamp for word: "{obj["word"]}". ' \
                          f'Fixing it by placing it after the previous word: "{previous.word}". At start: {vtd.start} end: {vtd.end}. Fix it manually!'
                print(f"{red_highlighted(msg)}")
            transcribed_data.append(vtd)  # and add it to list
    return transcribed_data
