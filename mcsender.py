#!/usr/bin/env python
""" mcsender.py
    Author: Bill Hallahan

    This file defines the MorseCodeSender class.

    Usage:
        python mcsender.py <text>
    
    <text>  The text to be convered to Morse code. If no text is specified,
            then the text "Lorum Ipsum" is used.

    Only the following methods of the MorseCodeSender class should be called
    by client code.

        get_words_per_minute
        set_words_per_minute
        get_tone_frequency
        set_tone_frequency
        shutdown
        wait_for_audio_to_complete
        stop
        send
"""

import os
import sys
import threading
from queue import Queue
import math
import wave
try:
    import sound
except:
    pass

class MorseCodeSender(threading.Thread):
    """ This class turns a text message into Morse code and plays the
        Morse code audio.
    """

    # The characters '~' and '%' have no Morse code equivalent.  These characters
    # have been added to the end of the Morse code dictionary to represent 'SK"
    # (end of message) and 'error' respectively.
    MORSE_SENDER_DICT = {
        'a' : '.-', 'b' : '-...', 'c' : '-.-.', 'd' : '-..', 
        'e' : '.', 'f' : '..-.', 'g' : '--.', 'h' : '....', 
        'i' : '..', 'j' : '.---', 'k' : '-.-', 'l' : '.-..', 
        'm' : '--', 'n' : '-.', 'o' : '---', 'p' : '.--.', 
        'q' : '--.-', 'r' : '.-.', 's' : '...', 't' : '-', 
        'u' : '..-', 'v' : '...-', 'w' : '.--', 'x' : '-..-', 
        'y' : '-.--', 'z' : '--..', '0' : '-----', '1' : '.----', 
        '2' : '..---', '3' : '...--', '4' : '....-', '5' : '.....', 
        '6' : '-....', '7' : '--...', '8' : '---..', '9' : '----.', 
        '.' : '.-.-.-', ',' : '--..--', '?' : '..--..', '-' : '-....-', 
        '/' : '-..-.', '(' : '-.--.', ')' : '-.--.-', '"' : '.-..-.', 
        "'" : '.----.', ':' : '---...', ';' : '-.-.-.', 
        '@' : '.--.-.', '$' : '...-..-', '=' : '-...-', '+' : '.-.-.', 
                         '~' : '.-.-.', '%' : '........' }

    def __init__(self,
                 words_per_minute=15.0,
                 tone_frequency=500.0,
                 sample_rate=11025,
                 audio_file_name='morse.wav'):
        """ Initialize this MorseCodeSender class instance. """
        self.words_per_minute = words_per_minute
        self.dot_time_in_msec = 1200.0 / self.words_per_minute
        self.tone_frequency = tone_frequency
        self.sample_rate = sample_rate
        self.sample_period = 1.0 / float(self.sample_rate)
        self.alpha = MorseCodeSender._get_alpha(self.sample_rate)
        self.one_minus_alpha = 1.0 - self.alpha
        self.audio_file_name = audio_file_name
        self.sample_buffer = None
        self.text_queue = Queue()
        self.dump_text_queue = False
        self.player = None
        self.audio_finished_event = threading.Event()
        self.audio_thread_continue = True
        threading.Thread.__init__(self)
        # The inherited threading.start() methods calls the derived
        # self.run() method in another thread.
        self.start()

    def __enter__(self):
        """ Method for startup using "with" statement. """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """ Method for cleanup using "with" statement. """ 
        self.shutdown()

    @staticmethod
    def _get_alpha(sample_rate):
        """ Set the tone envelope shaping parameter based on the sample rate.
            When alpha is used in the following equation, the value of
            'gain' starts at 0.0, and the equation is applied each sample,
            then 'gain' will increase to 0.5 in 2 milliseconds of samples.
                gain = (1.0 - alpha) + alpha * gain
        """
        alpha = 0.813900928542 + 0.0166631277223 * math.log(sample_rate)
        return alpha

    @staticmethod
    def _float_to_16_bit_sample(value):
        """ Convert a floating point value to a 16 bit signed value. """
        sample = int(32767.0 * value)
        byte0 = sample & 255
        byte1 = (sample >> 8) & 255
        return byte0, byte1

    def _synthesize_tone(self, duration_in_msec):
        """ Synthesize a tone for the specified duration and pitch frequency 
            at the specified sample rate.  The tone envelope is shaped
            to prevent clicks.
        """
        sample_count = int(float(self.sample_rate) * duration_in_msec * 0.001);
        # There are two bytes per 16-bit sample.
        tmp_buffer = bytearray(sample_count + sample_count)
        fscale = 2.0 * math.pi * self.tone_frequency * self.sample_period;
        # Make the leading edge of the tone pulse rise for 2 milliseconds
        # and the trailing edge fall for 2 milliseconds.
        leading_trailing_sample_count = int(float(self.sample_rate) * 0.002);
        middle_sample_count = sample_count - (2 * leading_trailing_sample_count)
        # Loop and create the audio samples.
        index = 0
        # Create the rising envelope part of the tone.
        # Use the value of 'gain' to shape the tone envelope.
        gain = 0.0
        for i in range(0, leading_trailing_sample_count):
            gain = self.one_minus_alpha + self.alpha * gain
            angle = float(i) * fscale
            value = 2.0 * gain * math.sin(angle)
            byte0, byte1 = MorseCodeSender._float_to_16_bit_sample(value)
            # Write the bytes in little-endian order.
            tmp_buffer[index] = byte0
            tmp_buffer[index + 1] = byte1
            index += 2
        # Create the level part of the tone. Start at the next
        # sample index so that the phase is a continuous function.
        temp_count = leading_trailing_sample_count + middle_sample_count;
        for i in range(leading_trailing_sample_count, temp_count):
            angle = float(i) * fscale
            value = math.sin(angle)
            byte0, byte1 = MorseCodeSender._float_to_16_bit_sample(value)
            # Write the bytes in little-endian order.
            tmp_buffer[index] = byte0
            tmp_buffer[index + 1] = byte1
            index += 2
        # Create the decaying envelope part of the tone.
        # Start at the next sample index so that the phase is a
        # continuous function.
        inv_gain = 0.0
        new_temp_count = temp_count + leading_trailing_sample_count;
        for i in range(temp_count, new_temp_count):
            inv_gain = self.one_minus_alpha + self.alpha * inv_gain
            gain = 0.5 - inv_gain
            angle = float(i) * fscale
            value = 2.0 * gain * math.sin(angle)
            byte0, byte1 = MorseCodeSender._float_to_16_bit_sample(value)
            # Write the bytes in little-endian order.
            tmp_buffer[index] = byte0
            tmp_buffer[index + 1] = byte1
            index += 2
        # Add the synthesized audio to the sample buffer.
        self.sample_buffer.extend(tmp_buffer)

    def _synthesize_silence(self, duration_in_msec):
        """ Synthesize a tone for the specified duration and pitch frequency 
            at the specified sample rate.  The tone envelope is shaped
            to prevent clicks.
        """
        if duration_in_msec > 0.0:
            sample_count = int(float(self.sample_rate) * duration_in_msec * 0.001);
            # There are two bytes per 16-bit sample.
            byte_count = sample_count + sample_count
            tmp_buffer = bytearray(byte_count)
            # Loop and create the audio samples.
            index = 0
            for i in range(0, byte_count):
                tmp_buffer[i] = 0
            # Add the synthesized audio to the sample buffer.
            self.sample_buffer.extend(tmp_buffer)

    def _create_morse_code_audio(self, text):
        """ Create an audio byte array. """
        # The Morse-sender-dictionary letter keys are lower-case letters.
        lctext = text.lower()
        # Replace any newline characters with a space character.
        lctext = lctext.replace('\n', ' ')
        # Loop and convert characters to Morse code audio.
        # All characters that are not in the Morse-sender-dictionary
        # and are not either a space or a tab character are discarded.
        silence_count = 0
        for c in lctext:
            if c in MorseCodeSender.MORSE_SENDER_DICT:
                code = MorseCodeSender.MORSE_SENDER_DICT[c]
                for dotdash in code:
                    if dotdash == '.':
                        # The symbol is a dot.
                        duration_in_msec = self.dot_time_in_msec
                    else:
                        # The symbol is a dash.
                        duration_in_msec = 3.0 * self.dot_time_in_msec
                    # Create a tone with the specified duration.
                    self._synthesize_tone(duration_in_msec)
                    # After each dot or dash, add one dot-duration of silence.
                    self._synthesize_silence(self.dot_time_in_msec)
                # After each character, add 2 more dot-durations of silence
                # resulting in three dot-durations of silence after a letter.
                self._synthesize_silence(2.0 * self.dot_time_in_msec)
                silence_count = 3
            else:
                # The letter is not in the Morse code dictionary. If the
                # letter is a space character or tab character, then make
                # sure there are 7 dot-durations of silence to create the
                # proper separation between words.
                if c == ' ' or c == '\t':
                    silence_length = 7 - silence_count
                    self._synthesize_silence(silence_length * self.dot_time_in_msec)
                    silence_count = 0

    @staticmethod
    def _delete_file(file_name):
        """ Delete a file. """
        try:
            os.remove(file_name)
        except:
            # The file did not exist.
            pass

    def _create_wave_file(self):
        """ Create a wave audio file with the specified data. """
        # If a wave file with the same name exists, then delete it.
        MorseCodeSender._delete_file(self.audio_file_name)
        # Create the wave file.
        is_wave_open = False
        try:
            wv = wave.open(self.audio_file_name, mode='wb')
            is_wave_open = True
            wv.setparams((1,  # 1 channel (mono)
                          2,  # 2 bytes per sample * 1 channel
                          self.sample_rate,
                          0,  # Initial number of samples.
                          'NONE',
                          'not compressed'))
            wv.writeframes(self.sample_buffer)
        except:
            print('Error creating audio file')
        if is_wave_open:
            wv.close()
        self.sample_buffer = None

    def _audio_finished_handler(self):
        """ Set in the sound.Player instance to indicate audio has completed.
        """
        # Indicate that no audio is playing. 
        self.audio_finished_event.set()
        # This check should not be necessary, but it cannot hurt.
        if self.player:
            self.player.finished_handler = None
            self.player = None

    def _wait_for_player_to_complete(self):
        # Wait for the audio to complete.
        self.audio_finished_event.wait()

    def _do_send(self, text):
        """ Primary function convert the text to Morse code audio
            and to play the audio.
        """
        # Free any existing bytearray object.
        if self.sample_buffer:
            self.sample_buffer = None
        # Create a bytearray object to store the audio data.
        self.sample_buffer = bytearray()
        # Convert the text to Morse code audio in a bytearray object.
        self._create_morse_code_audio(text)
        # Create the Morse code audio file.
        self._create_wave_file()
        # Play the Morse code audio file.
        self.player = sound.Player(self.audio_file_name)
        # Indicate that audio is playing. 
        self.audio_finished_event.clear()
        # The audio finished handler will set self.audio_finished_event
        # when the audio has completed playing.
        self.player.finished_handler = self._audio_finished_handler
        self.player.play()

    def run(self):
        """ The Morse code sending thread. Read text from the text queue and
            send the text using Morse code. This method, which is derived from
            the threading.Thread class, should be treated as a private method.
        """
        while self.audio_thread_continue:
            # Block in the "queue.Queue.get" method until text is written to
            # the queue.
            text = self.text_queue.get(True)
            try:
                if text and not self.dump_text_queue:
                    self._do_send(text)
                    self._wait_for_player_to_complete()
            finally:
                self.text_queue.task_done()

    def get_words_per_minute(self):
        """ Returns the current sending speed. """
        return self.words_per_minute

    def set_words_per_minute(self, words_per_minute):
        """ The sending speed, which must be in the range
            of 5.0 to 60.0 words per minute.
        """
        is_valid_wpm = 5.0 <= words_per_minute <= 60.0
        if is_valid_wpm:
            self.words_per_minute = words_per_minute
            self.dot_time_in_msec = 1200.0 / self.words_per_minute
        return is_valid_wpm

    def get_tone_frequency(self):
        """ Returns the current tone frequency. """
        return self.tone_frequency

    def set_tone_frequency(self, tone_frequency):
        """ The tone_frequency must be greater or equal to 200.0 Hz
            and less than half the sample frequency.
        """
        is_valid_tone_freq = \
            200.0 <= tone_frequency < (0.5 * float(self.sample_rate))
        if is_valid_tone_freq:
            self.tone_frequency = tone_frequency
        return is_valid_tone_freq

    def shutdown(self):
        """ Stop Morse code audio and shutdown the audio thread.
            The class instance can no longer be used after this
            method is called.
        """
        # Call self.stop() to stop audio playing and empty the queue.
        self.stop()
        # Clear the self.audio_thread_continue so the 'run' method
        # will exit, and the audio thread will then exit.
        self.audio_thread_continue = False
        # Unblock the self._wait_for_player_to_complete() method
        # in the run queue.
        self.audio_finished_event.set()
        # Write to the queue to unblock the call to Queue.get() in
        # the run method to allow the 'run' method to exit.
        self.text_queue.put('')
        # Wait for the 'run' thread to exit.
        self.join()

    def wait_for_audio_to_complete(self):
        """ Wait for all queued text to be played and
            all audio to complete.
        """
        # Wait for the text queue to drain in the "run" method.
        self.text_queue.join()
        # Wait for the last queued audio to complete.
        self._wait_for_player_to_complete()

    def stop(self):
        """ Stop Morse code audio. """
        # Disable text processing in the "run" method so
        # that the text queue can drain and become empty.
        self.dump_text_queue = True
        # If there is an audio player, stop audio from playing.
        if self.player:
            self.player.stop()
            # The player.stop method does not cause the
            # "_audio_finished_handler" method to be called.
            # Unblock the _wait_for_player_to_complete method.
            self.audio_finished_event.set()
        # Wait for the text queue to drain in the "run" method.
        self.text_queue.join()
        # When the queue items are dumped, a race condition with
        # the self.dump_text_queue flag could allow a single text
        # item to be processed, resulting in audio playing.
        # In case that happens, stop audio again.
        if self.player:
            self.player.stop()
            self.audio_finished_event.set()
        # Enable text processing again.
        self.dump_text_queue = False

    def send(self, text):
        """ Queue text to be played. """
        if text:
            self.text_queue.put(text)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = 'Lorum Ipsum'
    # If you don't want to use a "with" statement, this works too.
    #     mcsender = MorseCodeSender()
    # In that case, you should call mcsender.shutdown() method when
    # done with the class instance to cause the audio thread to exit.
    with MorseCodeSender() as mcsender:
        mcsender.send(text)
        mcsender.wait_for_audio_to_complete()
