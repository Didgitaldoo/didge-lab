# Transmission Line Modeling of Didgeridoos

DidgeLab's transmission-line model (`didgelab.sim.tlm_python`, `didgelab.sim.tlm_cython`) follows Frank Geipel's CADSD method (Geipel, 2003). CADSD splits the bore into a finite number of cylindrical and conical pieces and chains their acoustic transfer matrices, building on the transmission-line modelling of brass instruments (Caussé et al., 1984) and of conical and cylindrical horn elements (Mapes-Riordan, 1991/1993). The formulas below are those of the code; see the [References](#references).

## 1. The Core Concept: Transfer Matrices

The heart of the simulation is the Transfer Matrix (represented by the variable y in the ap function). For any segment of the instrument, the relationship between the input (mouthpiece side) and output (bell side) can be described as:

$$\begin{bmatrix} P_{in} \\ U_{in} \end{bmatrix} = \begin{bmatrix} A & B \\ C & D \end{bmatrix} \begin{bmatrix} P_{out} \\ U_{out} \end{bmatrix}$$

$P$ is Acoustic Pressure.

$U$ is Volume Velocity (flow).

The matrix $[A, B; C, D]$ encapsulates the geometry and physics of that specific segment.
The code iterates through all segments, performing matrix multiplication (z[0][0] = x[0][0] * y[0][0] + ...) to find the total transfer matrix for the entire instrument. The segment matrices for cylinders and cones are those of Mapes-Riordan (1993).

## 2. Viscothermal Losses

Sound doesn't travel perfectly in a tube; energy is lost due to friction against the walls (viscosity) and heat exchange (thermal conductivity).
The code accounts for this using the variable rvw (a dimensionless Reynolds-like number) and a complex propagation constant $T_w$ ($\Gamma$):

$$k_w = \frac{\omega}{c}$$

$$\Gamma \approx \frac{\omega}{c} \left( 1 + \frac{1.045}{\sqrt{w \cdot \text{geometry constants}}} \cdot (1 + i) \right)$$

This "complexifies" the wave number, meaning the sound wave both shifts in phase and decays in amplitude as it moves through the pipe. The loss terms are implemented as in CADSD (Geipel, 2003); the modelling of viscothermal losses in transmission-line models of wind instruments goes back to Caussé et al. (1984).

## 3. Conical vs. Cylindrical Geometry

The code handles both shapes inside the ap function's if (d0 != d1) block:
Cylindrical ($d_0 = d_1$): Uses standard hyperbolic sines and cosines ($cosh, sinh$).
Conical ($d_0 \neq d_1$): Uses a more complex derivation for spherical waves. It calculates $x_0$ and $x_1$, which represent the distances from the cone's virtual apex to the start and end of the segment.

## 4. Radiation Impedance ($Z_a$)

The Za function calculates what happens when the sound hits the end of the didgeridoo (the bell).
Sound doesn't just vanish; some of it reflects back into the tube. This "load" at the end of the line depends on the frequency and the diameter of the bell. DidgeLab uses the low-frequency limit of the Levine and Schwinger (1948) radiation impedance of an unflanged pipe with radius $a = d/2$ at the bell opening:

$$Z_{rad} \approx \frac{\rho c}{S}\left( \frac{(ka)^2}{4} + i \cdot 0.6\, ka \right) = \frac{\rho c}{S}\left( \frac{1}{16}\left(\frac{\omega d}{c}\right)^2 + i \cdot 0.3 \frac{\omega d}{c} \right), \qquad S = \frac{\pi d^2}{4},\ k = \frac{\omega}{c}.$$

The real part is the radiated sound power, the imaginary part an end correction: the bore acts as if it were $0.6\,a$ longer. (Up to version 3.0.x, DidgeLab used $\frac{1}{2} Z_c ((\omega d/c)^2 + i\,0.6\,L\,\omega d/c)$ after Geipel (2003), with $Z_c$ at the start of the last segment and $L$ its length; this shortened the end correction by the factor $L$ in metres. On the Didge Archive the Levine–Schwinger form predicts the open-hand slap resonances and the toots clearly better, see [synthesizer.md](../../synthesizer.md) §7.)

## 5. Input Impedance ($Z_e$)

Finally, the cadsd_Ze function calculates the Input Impedance at the mouthpiece. This is the value a player "feels" with their lips.

$$Z_{in} = \frac{A Z_a + B}{C Z_a + D}$$

Where $A, B, C, D$ are the components of the total chained matrix and $Z_a$ is the radiation impedance at the bell. The peaks in this impedance spectrum correspond to the resonant frequencies (the notes) of the didgeridoo.

## References

- Caussé, R., Kergomard, J., Lurton, X. (1984): *Input impedance of brass musical instruments — Comparison between experiment and numerical models*. J. Acoust. Soc. Am. 75(1), 241–254.
- Geipel, F. (2003): *CADSD – Computer Aided Didgeridoo Sound Design*. [didgeridoo-physik.de/cadsd/cadsd-method](https://www.didgeridoo-physik.de/cadsd/cadsd-method/)
- Levine, H., Schwinger, J. (1948): *On the radiation of sound from an unflanged circular pipe*. Phys. Rev. 73(4), 383–406.
- Mapes-Riordan, D. (1993): *Horn modeling with conical and cylindrical transmission-line elements*. J. Audio Eng. Soc. 41(6), 471–482. First presented at the 91st AES Convention (1991).
