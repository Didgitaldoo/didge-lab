# The `.didge` song format

A song is a plain text file with a **header** and up to two **lanes**,
`[rhythm]` and `[tongue]`, written in bars. Example
([songs/dong-tiki-ritititi.didge](../src/didgelab/synth/songs/dong-tiki-ritititi.didge)):

```
# comments start with #
title:      Dong-tiki ritititi
instrument: Arusha
tempo:      92
meter:      4/4

[rhythm]
| !dong-tiki dong-tiki dong:1/8 tiki:1/8 ritititi |
```

Render it:

```bash
didgelab-synth play src/didgelab/synth/songs/dong-tiki-ritititi.didge                 # one shot  -> dong-tiki-ritititi.wav
didgelab-synth play src/didgelab/synth/songs/dong-tiki-ritititi.didge --loop 4        # seamless loop of 4 passes
didgelab-synth play src/didgelab/synth/songs/dong-tiki-ritititi.didge --repeat 4      # one shot, played 4 times
didgelab-synth play src/didgelab/synth/songs/dong-tiki-ritititi.didge --tempo 120 -o fast.wav
didgelab-synth play src/didgelab/synth/songs/dong-tiki-ritititi.didge --list          # print the timed syllables
didgelab-synth play src/didgelab/synth/songs/dong-tiki-ritititi.didge --play          # also play it (needs sounddevice)
```

## Header

| field | meaning | default |
|---|---|---|
| `title` | name of the song | file name |
| `instrument` | Didge Archive name (`Arusha`, `Tamaki1`, …) or path to a geometry JSON (`[[x_mm, d_mm], …]`, relative to the song file) | `Arusha` |
| `tempo` | quarter notes per minute; `--tempo` overrides it | 100 |
| `meter` | e.g. `4/4`, `3/4`, `7/8` | `4/4` |
| `swing` | position of the second eighth within a beat, `50%` = straight, `66%` = triplet feel | 50% |
| `humanize` | timing, pitch and mouth micro-variation, 0 … 1 | 1 |
| `glide` | time constant of mouth-shape changes in seconds | 0.05 |
| `voice` | default pitch of vocalised syllables (`~`) | A3 |
| `brightness` | lip brightness, 0 (dull) … 1 (bright) | 0.56 |
| `breath` | breath noise in dB relative to the tone | −30 |
| `reverb` | room level, 0 … 1 | 0.25 |
| `seed` | random seed of the humanisation | 0 |

## Bars and timing

Bars are separated by `|` and can span several lines. **A bar is divided
equally among its tokens**:

```
| doo doo doo doo |                  four quarters
| du du du du du du du du |          eight eighths
| doo:1/2 du du |                    a half note and two quarters
```

A token with an explicit length (`:1/8` = eighth note, `:3/16`, `:1/2`, or
`:2` = two beats) takes that length; the tokens without one share the rest of
the bar. The lengths must fill the bar exactly.

**Inside a token**, hyphens split its time equally, and the syllables of each
part share that part equally:

| token (one beat) | result |
|---|---|
| `dong-ti` | dong (1/8), ti (1/8) |
| `tiki` | ti (1/16), ki (1/16) … with `tiki du` filling a beat |
| `dong-tiki` | dong (1/8), ti (1/16), ki (1/16) |
| `ritititi` | ri, ti, ti, ti (1/16 each) |
| `di-ga-di` | three triplet eighths |

## Rhythm lane

A **syllable** is `[consonant] vowel [ending]`:

| part | options | sound |
|---|---|---|
| consonant | `t` `d` | tongue tip closes at the alveolar ridge (t harder than d) |
| | `k` `g` | back of the tongue closes at the velum |
| | `p` | lip/cheek pulse |
| | `r` | one quick tongue flap |
| | `rr` | rolled r (trill) over the whole syllable |
| | `y` `w` | glide from "ee" / "oo" |
| | `h` `ch` `s` | aspiration / fricatives |
| vowel | `ee`/`i`, `e`, `ae`, `a`, `o`, `oo`/`u`, `uh` | mouth shape from tongue front (`ee`) to back (`oo`); `uh` is the relaxed, open drone |
| ending | `ng` | tongue moves to the velum at the end (*dong*) |
| | `n` | tongue tip moves up at the end |

Other tokens:

| token | meaning |
|---|---|
| `.` | rest (the lips stop) |
| `_` | hold the previous syllable for this slot (ties across bars too) |
| `^` | circular breathing |
| `!dong` | accent |
| `*too`, `**too` | toot on the 2nd / 3rd resonance |
| `~oo`, `~oo@C4` | vocalise (sing into the didgeridoo), pitch from `@` or the `voice` header |

Prefixes apply to the whole token; `!` accents its first syllable.

## Tongue lane

The tongue lane is optional. It overrides the mouth shape of the syllables
and uses the same bar and length rules:

| token | meaning |
|---|---|
| `ee` `e` `ae` `a` `o` `oo` | tongue (mouth shape) of that vowel, held during the slot |
| `front` `mid` `back`, `0` … `1` | position on the tongue axis, 0 = front, 1 = back |
| `~` | glide: the tongue moves linearly from the end of the previous value to the start of the next value |
| `.` | keep holding the previous value; the next value follows quickly (with the `glide` time) |
| `auto` / `*` | no override: the syllables decide |

```
| oo ~ ~ ee |     hold "oo" for a beat, glide for two beats, "ee" on beat four
| oo . . ee |     "oo" for three beats, then a quick change to "ee"
| oo ~ ~ ~ | ~ ~ ~ ee |   a slow sweep over two bars
| auto | oo ~ ee . |      bar 1 follows the syllables, bar 2 is automated
```

If the tongue lane is shorter than the rhythm lane, the remaining bars follow
the syllables. Stop consonants still close the tongue briefly on top of the
automation.

## Loops

`--loop N` renders a seamless loop of `N` passes: the song is played `N + 2`
times in one continuous performance, the middle `N` passes are cut out, and
the 30 ms that follow the cut are crossfaded into its beginning. The file is
exactly `N` × bar length × bars long and loops without a click. Without
`--loop` the song is a one shot: the lips start at the beginning and stop at
the end, and bore and room ring out for one second.

## Example songs

| file | content |
|---|---|
| [dong-ti.didge](../src/didgelab/synth/songs/dong-ti.didge) | dong-ti ×4 |
| [tiki-du.didge](../src/didgelab/synth/songs/tiki-du.didge) | tiki du ×4 |
| [dong-tiki-ritititi.didge](../src/didgelab/synth/songs/dong-tiki-ritititi.didge) | dong-tiki dong-tiki dong tiki ritititi |
| [drone-back-to-front.didge](../src/didgelab/synth/songs/drone-back-to-front.didge) | four-bar drone, tongue from back to front |
| [groove-demo.didge](../src/didgelab/synth/songs/groove-demo.didge) | swing, tongue automation, circular breathing, toot, vocalisation |

Rendered versions are written by `python scripts/synth/make_figures.py` to `outputs/synth-audio/songs/`
(one shot and `-loop` with 4 passes). These MP3s are for listening only: MP3
encoders add a few milliseconds of silence, so only the WAV files written by
`didgelab-synth play --loop` loop seamlessly.
