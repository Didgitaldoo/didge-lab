import math

# note number to frequency (e.g. 0 -> 440)
def note_to_freq(note):
    return 440*pow(2, note/12)


# note number to name (e.g. 0->A4)
def note_name(note):
    note=round(note)
    note+=48
    octave=math.floor(note/12)
    number=note%12
    notes=["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    name=str(notes[number]) + str(octave)
    return name


# frequency to note number (e.g. 440 -> 0)
def freq_to_note(freq):
    return 12* (math.log2(freq) - math.log2(440))


# frequency to note and cent difference from that note (e.g. 440 -> 0,0)
def freq_to_note_and_cent(freq):
    
    note_fuzzy=freq_to_note(freq)
    note=round(note_fuzzy)
    diff=note-note_fuzzy
    return note, diff*100



# get wavelength of soundwave with frequency freq in mm
def freq_to_wavelength(freq):
    c=343.2
    return 1000*c/freq

# note name to number, e.g. A4 -> 440
def note_name_to_number(note):
    assert len(note) in (2, 3)
    if len(note) == 3:
        assert note[1] == "#"

    names = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    octave = int(note[-1])
    n = names.index(note[0:len(note) - 1])

    return 12 * octave + n + - 48

