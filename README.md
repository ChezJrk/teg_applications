# Teg Applications
This is a research artifact for the paper: **Systematically Optimizing Parametric Discontinuities**. This repository contains the implementation for the applications of our differentiable programming language Teg. The applications include image stylization, fitting shader parameters, trajectory optimization, and optimizing physical designs. The implementation for Teg can be found at [https://github.com/ChezJrk/Teg](https://github.com/ChezJrk/Teg).

## Setup
To install the necessary Python packages run:
```
pip install -r requirements.txt
```

## Optimizing Physical Design
We study optimization of discontinuous parameters that control a physical design.
We consider an idealized series of bungee cords that deform.
After deformation, a string prevents further extension of the spring.
We jointly minimize the time and acceleration of a person connected to this bungee-string system.
We optimize the spring constants $k_1, k_2$ and the lengths of the strings $l_1, l_2$. 
We add the hard constraint that the person does not hit the ground to prevent death.

To optimize using derivatives computed by Teg run:
```
python3 physics/springs.py
```

To optimize using Teg, but ignoring the derivative contribution from the discontinuities (as would be done in PyTorch, TensorFlow, etc.) run:
```
python3 physics/springs.py --ignore_deltas
```

To optimize using derivatives estimated using finite differences run:
```
python3 physics/springs.py --finite_diff
```

Running each of these should take more than a few minutes.