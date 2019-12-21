#!/usr/bin/env python
#=======================================================================
# Copyright (C) 2017 William Hallahan
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#=======================================================================
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
import Queue
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
    # have been added to the end of the Morse code dictionary to represent 'SK'
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
        '~' : '...-.-', '%' : '........' }

    def __init__(self,
                 words_per_minute=15.0,
                 tone_frequency=500.0,
                 sample_rate=11025,
                 audio_file_name='morse.wav'):
        """ Initialize this MorseCodeSender class instance. """
        self.words_per_minute = words_per_minute
        self.dot_time_in_msec = 0.0
        self.tone_frequency = tone_frequency
        self.sample_rate = sample_rate
        self.sample_period = 1.0 / float(self.sample_rate)
        self.audio_file_name = audio_file_name
        # Buffers to cache synthesized sample data.
        self.pulse_shaping_list = []
        self.dot_sample_buffer = None
        self.dash_sample_buffer = None
        self.silence_4_sample_buffer = None
        self.silence_2_sample_buffer = None
        self.silence_1_sample_buffer = None
        # The main sample buffer.
        self.sample_buffer = None
        # Text queue data.
        self.text_queue = Queue.Queue()
        self.stop_and_clear_queue = False
        # Set the dot time in milliseconds based on the sending speed.
        self.set_words_per_minute(self.words_per_minute)
        # Initialize the sample buffers.
        self._cache_dot_dash_sample_data()
        self._cache_silence_sample_data()
        # Audio data.
        self.player = None
        self.audio_finished_event = threading.Event()
        self.audio_thread_continue = True
        threading.Thread.__init__(self)
        # The inherited threading.start() methods calls the
        # derived self.run() method in another thread.
        self.start()

    def __enter__(self):
        """ Method for startup using "with" statement. """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """ Method for cleanup using "with" statement. """ 
        self.shutdown()

    @staticmethod
    def _float_to_16_bit_sample(value):
        """ Convert a floating point value to a 16 bit signed value. """
        sample = int(32767.0 * value)
        byte0 = sample & 255
        byte1 = (sample >> 8) & 255
        return byte0, byte1

    def _get_pulse_shaping_waveform(self):
        """ Set the pulse shaping envelope based on the sample rate
            and dot time.
        """
        self.pulse_shaping_list = []
        # Make the rise time be 3.3333% if the dot time.
        rise_time_in_msec = 0.03333333333333 * self.dot_time_in_msec
        # Limit the rise time to 2 milliseconds.
        if rise_time_in_msec > 0.002:
            rise_time_in_msec = 0.002
        rising_falling_count = int(rise_time_in_msec * self.sample_rate)
        step = math.pi / rising_falling_count
        # The first value is zero, so skip that value.
        # The last value is 1.0, so skip that value too.
        for i in range(1, rising_falling_count - 1):
            gain = 0.5 * (1.0 - math.cos(step * i))
            self.pulse_shaping_list.append(gain)

    def _synthesize_tone(self, duration_in_msec):
        """ Synthesize a tone pulse for the specified duration and the
            specified pitch frequency at the specified sample rate.  The
            tone envelope is shaped to prevent clicks.
        """
        sample_count = int(float(self.sample_rate) * duration_in_msec * 0.001)
        # There are two bytes per 16-bit sample.
        tmp_buffer = bytearray(sample_count + sample_count)
        fscale = 2.0 * math.pi * self.tone_frequency * self.sample_period;
        # Loop and create the audio samples.
        index = 0
        # Create the rising envelope part of the tone.
        for i, gain in enumerate(self.pulse_shaping_list):
            angle = float(i) * fscale
            value = gain * math.sin(angle)
            byte0, byte1 = MorseCodeSender._float_to_16_bit_sample(value)
            # Write the bytes in little-endian order.
            tmp_buffer[index] = byte0
            tmp_buffer[index + 1] = byte1
            index += 2
        # Create the level part of the tone. Start at the next
        # sample index so that the phase is a continuous function.
        rising_falling_count = len(self.pulse_shaping_list)
        middle_sample_count = sample_count - (2 * rising_falling_count)
        for i in range(0, middle_sample_count):
            angle = float(i + rising_falling_count) * fscale
            value = math.sin(angle)
            byte0, byte1 = MorseCodeSender._float_to_16_bit_sample(value)
            # Write the bytes in little-endian order.
            tmp_buffer[index] = byte0
            tmp_buffer[index + 1] = byte1
            index += 2
        # Create the decaying part of the tone. Start at the next
        # sample index so that the phase is a continuous function.
        temp_count = rising_falling_count + middle_sample_count;
        for i, rev_gain in enumerate(self.pulse_shaping_list):
            angle = float(i + temp_count) * fscale
            value = (1.0 - rev_gain) * math.sin(angle)
            byte0, byte1 = MorseCodeSender._float_to_16_bit_sample(value)
            # Write the bytes in little-endian order.
            tmp_buffer[index] = byte0
            tmp_buffer[index + 1] = byte1
            index += 2
        return tmp_buffer

    def _synthesize_silence(self, duration_in_msec):
        """ Synthesize silence for the specified duration at the specified
            sample rate.
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
        return tmp_buffer

    def _cache_dot_dash_sample_data(self):
        """ Caches a dot tone pulse and a dash tone pulse. """
        self._get_pulse_shaping_waveform()
        self.dot_sample_buffer = self._synthesize_tone(self.dot_time_in_msec)
        self.dash_sample_buffer = self._synthesize_tone(3.0 * self.dot_time_in_msec)

    def _cache_silence_sample_data(self):
        """ Caches silence data for three durations. """
        self.silence_4_sample_buffer = self._synthesize_silence(4.0 * self.dot_time_in_msec)
        self.silence_2_sample_buffer = self._synthesize_silence(2.0 * self.dot_time_in_msec)
        self.silence_1_sample_buffer = self._synthesize_silence(self.dot_time_in_msec)

    def _create_morse_code_audio(self, text):
        """ Create an audio byte array for the passed text. """
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
                        self.sample_buffer.extend(self.dot_sample_buffer)
                    else:
                        # The symbol is a dash.
                        self.sample_buffer.extend(self.dash_sample_buffer)
                    # After each dot or dash, add one dot-duration of silence.
                    self.sample_buffer.extend(self.silence_1_sample_buffer)
                # After each character, add 2 more dot-durations of silence
                # resulting in three dot-durations of silence after a letter.
                self.sample_buffer.extend(self.silence_2_sample_buffer)
                silence_count = 3
            else:
                # The letter is not in the Morse code dictionary. If the
                # letter is a space character or tab character, then make
                # sure there are 7 dot-durations of silence to create the
                # proper separation between words.
                if c == ' ' or c == '\t':
                    silence_length = 7 - silence_count
                    if silence_length > 3:
                        self.sample_buffer.extend(self.silence_4_sample_buffer)
                        silence_length -= 4
                    if silence_length > 1:
                        self.sample_buffer.extend(self.silence_2_sample_buffer)
                        silence_length -= 2
                    if silence_length > 0:
                        self.sample_buffer.extend(self.silence_1_sample_buffer)
                        silence_length -= 1
                    silence_count = 0

    def _create_wave_file(self):
        """ Create an wave audio file that contains the audio data in
            bytearray 'self.sample_buffer'.
        """
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

    def _audio_finished_handler(self):
        """ Set in the sound.Player instance to indicate audio has completed.
        """
        # Indicate that no audio is playing. 
        self.audio_finished_event.set()
        self.player = None

    def _wait_for_player_to_complete(self):
        # Wait for the audio to complete.
        self.audio_finished_event.wait()

    def _do_send(self, text):
        """ Primary function convert the text to Morse code audio
            and to play the audio.
        """
        self.sample_buffer = bytearray()
        # Fill self.sample_buffer with audio samples.
        self._create_morse_code_audio(text)
        # Write self.sample_buffer data to a wave audio file.
        self._create_wave_file()
        self.sample_buffer = None
        # Play the Morse code audio file.
        self.audio_finished_event.clear()
        self.player = sound.Player(self.audio_file_name)
        self.player.finished_handler = self._audio_finished_handler
        self.player.play()

    def run(self):
        """ The Morse code sending thread. Read text from the text queue and
            send the text using Morse code. This method, which is derived from
            the threading.Thread class, should be treated as a private method.
        """
        while self.audio_thread_continue:
            # Block in the "Queue.Queue.get" method until text
            # is written to the queue.
            text = self.text_queue.get(True)
            if not self.stop_and_clear_queue:
                try:
                    if text:
                        self._do_send(text)
                        self._wait_for_player_to_complete()
                finally:
                    self.text_queue.task_done()
            else:
                self.text_queue.task_done()
                # Stop audio from playing.
                if self.player:
                    self.player.stop()
                    self.player = None
                # Empty the text queue.
                try:
                    while True:
                        self.text_queue.get(False)
                        self.text_queue.task_done()
                except Queue.Empty:
                    pass

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
        # Synthesizes sample data for the current dot length.
        self._cache_dot_dash_sample_data()
        self._cache_silence_sample_data()
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
        # Synthesizes sample data for the current tone frequency.
        self._cache_dot_dash_sample_data()
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
        # Unblock the self._wait_for_player_to_complete() method
        # in the run queue again.
        self.audio_finished_event.set()
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
        # Set a flag to stop audio and clear the queue.
        self.stop_and_clear_queue = True
        # Unblock the _wait_for_player_to_complete()
        # in the "run" method.
        self.audio_finished_event.set()
        # It's possible that immediately after the audio
        # finished event is set, the event cleared in "run"
        # thread inside of the _do_send method. To synchronize
        # the shutdown, send an empty test string to the
        # queue and then set the audio finished event again.
        self.text_queue.put('')
        self.audio_finished_event.set()
        # Wait for the text queue to become empty.
        self.text_queue.join()
        # Enable text processing again.
        self.stop_and_clear_queue = False

    def send(self, text):
        """ Queue text to be played. """
        if text:
            self.text_queue.put(text)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = 'hello world'
    # If you don't want to use a "with" statement, this works too.
    #     mcsender = MorseCodeSender()
    # In that case, you should call mcsender.shutdown() method when
    # done with the class instance to cause the audio thread to exit.
    with MorseCodeSender() as mcsender:
        mcsender.send(text)
        mcsender.wait_for_audio_to_complete()
