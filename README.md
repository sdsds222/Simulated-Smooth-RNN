# Simulated-Smooth-RNN
recurrent memory cell for sequence processing. Its core innovation provides flexible, high-resolution memory addressing at a constant O(1) cost. It achieves this with a controller that generates continuous floating-point addresses (e.g., 50.6), which are resolved by differentiable linear interpolation using only the two nearest slots.
