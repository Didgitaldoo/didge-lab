# Transmission Line Modeling of Didgeridoos

1. The Core Concept: Transfer Matrices
The heart of the simulation is the Transfer Matrix (represented by the variable y in the ap function). For any segment of the instrument, the relationship between the input (mouthpiece side) and output (bell side) can be described as:
$$\begin{bmatrix} P_{in} \\ U_{in} \end{bmatrix} = \begin{bmatrix} A & B \\ C & D \end{bmatrix} \begin{bmatrix} P_{out} \\ U_{out} \end{bmatrix}$$
$P$ is Acoustic Pressure.
$U$ is Volume Velocity (flow).
The matrix $[A, B; C, D]$ encapsulates the geometry and physics of that specific segment.
The code iterates through all segments, performing matrix multiplication (z[0][0] = x[0][0] * y[0][0] + ...) to find the total transfer matrix for the entire instrument.

2. Viscothermal Losses
Sound doesn't travel perfectly in a tube; energy is lost due to friction against the walls (viscosity) and heat exchange (thermal conductivity).
The code accounts for this using the variable rvw (a dimensionless Reynolds-like number) and a complex propagation constant $T_w$ ($\Gamma$):
$$k_w = \frac{\omega}{c}$$
$$\Gamma \approx \frac{\omega}{c} \left( 1 + \frac{1.045}{\sqrt{w \cdot \text{geometry constants}}} \cdot (1 + i) \right)$$
This "complexifies" the wave number, meaning the sound wave both shifts in phase and decays in amplitude as it moves through the pipe.

3. Conical vs. Cylindrical Geometry
The code handles both shapes inside the ap function's if (d0 != d1) block:
Cylindrical ($d_0 = d_1$): Uses standard hyperbolic sines and cosines ($cosh, sinh$).
Conical ($d_0 \neq d_1$): Uses a more complex derivation for spherical waves. It calculates $x_0$ and $x_1$, which represent the distances from the cone's virtual apex to the start and end of the segment.

4. Radiation Impedance ($Z_a$)
The Za function calculates what happens when the sound hits the end of the didgeridoo (the bell).
Sound doesn't just vanish; some of it reflects back into the tube. This "load" at the end of the line depends on the frequency and the diameter of the bell. The formula used here is a variation of the Levine and Schwinger radiation model, simplified by Geipel:
$$Z_{rad} \approx \frac{1}{2} Z_c \left( \left(\frac{\omega d}{c}\right)^2 + i \cdot 0.6 \frac{\omega d}{c} \right)$$

5. Input Impedance ($Z_e$)
Finally, the cadsd_Ze function calculates the Input Impedance at the mouthpiece. This is the value a player "feels" with their lips.
$$Z_{in} = \frac{A Z_a + B}{C Z_a + D}$$
Where $A, B, C, D$ are the components of the total chained matrix and $Z_a$ is the radiation impedance at the bell. The peaks in this impedance spectrum correspond to the resonant frequencies (the notes) of the didgeridoo.



