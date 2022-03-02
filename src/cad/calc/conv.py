import math

def note_to_freq(note):
    return 440*pow(2, note/12)

def note_name(note):
    note=round(note)
    note+=48
    octave=math.floor(note/12)
    number=note%12
    notes=["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    name=str(notes[number]) + str(octave)
    return name

def freq_to_note(freq):
    return 12* (math.log2(freq) - math.log2(440))

def freq_to_note_and_cent(freq):
    
    note_fuzzy=freq_to_note(freq)
    note=round(note_fuzzy)
    diff=note-note_fuzzy
    return note, diff*100

# get wavelength of soundwave with frequency freq in mm
def freq_to_wavelength(freq):
    c=343.2
    return 1000*c/freq