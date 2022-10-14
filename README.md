# DidgeLab

## 1. Introduction

DidgeLab is a free toolkit to compute didgeridoo geometries. Traditionally, building a didgeridoo is a random process. Builders know how the geometry influences the sound, but the exact sonic properties of the didgeridoo can only be determined after it was built. The DidgeLab software helps didgeridoo builders to first define the sound and then compute the according geometry.

It mainly can do two things:

1. Acoustical simulation to compute resonant frequencies of a didgeridoo geometry. This functionality is very similar to [Didgmo](https://sourceforge.net/projects/didgmo/) and [DidjiImp](https://sourceforge.net/projects/didjimp/).

2. Computational evolution to find didgeridoo shapes with certain sonic properties. This functionality is inspired by the works of [Frank Geipel](https://www.didgeridoo-physik.de/).

So the first functionality takes a didgeridoo geometry as input and computes its sonic properties. The 2nd functionality works vice versa: It takes sonic properties as an input and generates a didgeridoo geometry with these sonic capabilities. Especially the 2nd functionality is super interesting. To the best of my knowledge, DidgeLab is the first open toolkit to implement this functionality.

Unfortunately the software is complicated to use. It is a command line application, which means: There is no graphical user interface. When you use the software you most likely need to program in python, because it is more of a python programming toolkit than a normal end user software.

If you do not want to use DidgeLab itself, you can still use the geometry library (see below), which publishes some Didgeridoo geometries computed with DidgeLab.

## 2. Related works

DidgeLab stands on the shoulders of giants:

* Most important, the [pioneering works of Frank Geipel]((https://www.didgeridoo-physik.de/)) and his Computer-Aided-Dideridoo-Sound-Design-Tools (CADSD). I am not exactly sure what Frank Geipel did. But after reading through his website 1000 times I implemented DidgeLab in the way I understood he implemented it.
* The acoustic simulation of the didgeridoo is based on transmission line modeling, see [Dan Mapes-Riordan, 1991: Horn Modeling with Conical and Cylindrical Transmission Line Elements
](https://www.aes.org/e-lib/browse.cfm?elib=5522).
* There is important background information on didgeridoo acoustics in [Andrea Ferronis Youtube channel](https://www.youtube.com/) Andrea also offers webinars on Didgeridoo which are super informative. Most notable Youtube videos are:
  * [DIDGMO - The didgeridoo sound design software. How to set it and how to read the results
](https://www.youtube.com/watch?v=wWRTe3FMCWI&t=1017s)
  * [Acoustic and color tone of cylindrical and conical didgeridoo](https://www.youtube.com/watch?v=idcPw0RaUiE)
  * [Didgeridoo backpressure - Everything you know about it is probably wrong](https://www.youtube.com/watch?v=2q5TCpuf07U)
  

## 3. What can it do? Where are practical examples?

Well. Thats the thing. Currently I never built a Didgeridoo with a shape that this software computed. I computed some shapes with interesting sonic properties. I also spent a month in a 3d printing lab but I invested a lot of time in the printer and could not get it to work. After a pause from the 3d printer, I will restart this effort. So meanwhile I release this software as open source.

But the software should be able to do these things:

* You first think of the sound the didgeridoo should produce. And then the software computes a shape with that sound.
* Compute didgeridoos with precisely tuned resonant frequencies. So if you want a Didgeridoo that, for example, has a drone in D and toots in F, G and B, the software can produce a shape with these precisely tuned toots.
* Produce [singing](https://www.didgeridoo-physik.de/didge-physics/singer/) didgeridoos with super dominant overtones.
* Produce shapes that hardly exist in nature, of that come up very seldomly. Frank Geipel did this with the [Long Multi Tooter](https://www.didgeridoo-physik.de/didge-physics/long-multi-tooter/) but there might be more forms out there.
* And more. You can check out [Frank Geipels](https://www.didgeridoo-physik.de/news-archiv/) blog. With some extensions, DidgeLab should be able to do similar things.

You might ask yourself how I verify that the software works if I never built a didge with it. The Didgmo and Didjimp programs are known to work. If I put didgeridoo geometries computed by DidgeLab into these programs, they produce the same impedance spektra that I computed with DidgeLab. But of course I would love to build some Didgeridoos and it will not take long until I have a better verification.

## 4. Usage

### 4.1 Installation

Up to now it ran solely on Ubuntu linux. It should run in MS Windows and MacOS too, but nobody every tried that.

Prerequisites

* python 3.8 (other python versions will work also, but nobody tried yet)
* we recommend a virtual environment using e.g. pip or conda. In this environment, you install all required python packages with pip:

```
pip install -r requirements.txt
```

To use the didge reporting tool you need to have latex installed. Latex distributions tend to be large, e.g. tex-live is 5 GB. So you can start without Latex and then install it when you want to use the didge-reporting tool. Assuming you are using Ubuntu Linux, use this command. Sorry I did not test this because i have latex already installed. Probably a smaller Latex distribution will work also.

```
apt install texlive-full
```

### 4.2 Documentation

There is a tutorial series how to use the code. The tutorials also explain on how DidgeLab works.

* [Tutorial 1 - Introduction and acoustical simulation](doc/tutorials/tutorial1.ipynb)
* [Tutorial 2 - Shapes and Parameters](doc/tutorials/tutorial2.ipynb)
* [Tutorial 3 - Computational Evolution](doc/tutorials/tutorial3.ipynb)
* [Tutorial 4 - Advanced Concepts](doc/tutorials/tutorial4.ipynb)

If you use the system and struggle because you need more information, then please open a GitHub issue. I can write more documentation if there is a demand for it.

### 4.3 How to build a didgeridoo after these specifications.

This is out of scope for this document. Again I refer to [Frank Geipels Website](https://www.didgeridoo-physik.de), he wrote down some ideas. I am trying 3d printing but there are other ways to do it.

## 5. Didgeridoo geometry library

There is a series of didgeridoo geometries with interesting sonic capabilities published as well. If you have a Didgeridoo geometry with interesting sonic properties you can publish it here also. The Didgeridoo shapes are free for non-commercial use. For commercial use, please contact me.

Geometries are published as a series of segments that describe the inner shape of the instrument. Here is an example with three segments:

```
0 32
1000 40
1200 60
```

Each segment consists of two coordinates: The first coordinate describes the distance from the mouthpiece to the segment in mm. The 2nd coordinate describes the diameter of the bore at this coordinate. So the didgeridoo in this example has length 1200mm and a bell size of 60mm.

These are the geometries:

* [Arusha 1](geometries/arusha_1/arusha_1.md)
* [Arusha 2](geometries/arusha_2/arusha_2.md)
* [Arusha 3](geometries/arusha_3/arusha_3.md)
* [Kizimkazi](geometries/kizimkazi/kizimkazi.md)
* [Matema](geometries/matema/matema.md)

## 6. Get involved

This project is open source. You are most welcome to contribute to it. Please use the GitHub issues system for questions, feedback, etc. You can send me a private message on GitHub, but please consider: If the information might be interesting for others, which is especially the case for questions or bug reports, then use the GitHub issues system so others can see this conversation also.

It would be super great if you published the didgeridoo geometries you create with DidgeLab here in the geometry library. Knowledge should be shared instead of being kept secret.

## 7. Licensing / Hoch much does it cost?

The software is free and released under the GNU GPL v2.0 license.

The Didgeridoo shapes are free for non-commercial use.

## 8. Future works

The current version is certainly not the end. There are many ways to further develop this software. Here are some ideas for future directions.

### 8.1 Building Didgeridoos

Most importantly, I want to use this software to examine further didgeridoo shapes. I am super interested in reproducing and understanding Frank Geipels work. Another interesting thing are wet / drop octave didgeridoos which I would like to further explore with this software. 

### 8.2 Faster acoustical simulation

One big drawback of this software is that even with a beefy computer (e.g. with 40 CPU cores), the evolution takes a long time, e.g. one day. I have some ideas to make it faster:

1. Use clever shapes that leave out geometries that would not work anyways.
2. Optimize the acoustical simulation. The acoustical simulation computes the impedance for a given frequency. To compute a spektrum it computes e.g. all impedances from 30-1000 Hz. The low octave C1-C2 spans ~65 Hz, which means we need a high resolution to get a precise tuning. A higher octave, e.g. C3-C4, spans ~262 Hz, so for the same tuning precision as for the lower octave we can use a lower resolution. Right now, DidgeLab computes frequency spektra from 30-100 Hz with 0.1 Hz resolution (which runs 700 acoustical simulations) and from 100-1000 Hz with 1 Hz resolution (which runs 900 simulations). It should not be too difficult to find a more clever way to get the same tuning precision over all octaves and safe acoustical simulations.
3. Also, depending on the loss function, one can save a lot of computing power when we skip certain frequencies, e.g. frequencies below the drone frequency. If the drone is at D1 / 73.41 Hz and we skip everything below 70 Hz, this would safe 400 acoustical simulations, which saves 25% of the computing power. 

### 8.3 Gradient descent instead of computational evolution

Use a gradient based optimization instead of computational evolution. There is already an implementation for the acoustical simulation in PyTorch in src/cad/cadsd/pytorch. The idea is to use PyTorchs automatic gradient computation to save us from computing the gradient "by hand".

### 8.4 Graphical user interface

Implementing a graphical user interface to make the software easier to use for non-programmers. For me, personally, this is the least interesting, but for others it is probably super interesting.
