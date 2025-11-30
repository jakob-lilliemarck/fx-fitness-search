## Regurgitating nonsense
I've spent the better part of this Autumn thinking about and experimenting with machine learning in Rust. Initially I was curious if it would be feasible for me to do, and what it would take in terms of computational, not to mention cognitive resources - would it be within my reach?

"Why Rust?", you may ask. Why not Python, or even C? I have been asking myself the same question, but truth be told, I find neither of which compelling to work with. After all, in most systems (yes, even machine learning ones) the majority of the code is not about tensors and models, but about data retrieval and preprocessing. If you ask me, Rust has it all - excellent support for test driven development, memory safety, robust and explicit parallel computing and a proper expressive type system. I'd even argue that Rust is a simple language because in Rust there are no surprises and there is no magic.

## The Morphological Space of Neural Networks
I've learned that while doing machine learning, there are a lot of decisions to make. How to handle missing data? How to process, extract and structure data in a form the models and frameworks will accept? Which model, architecture and training parameters to use, and for which use-case? The list goes on.

From memory, most examples or articles I've seen about machine learning (though I must admit I really haven't done my homework here) typically involve an enthusiastic author reasoning about the validity and meaning of parameters while making one-off experiments. That makes me itch. If there's any equivalent to "code smell" for data science, I think that's it—you've got these sophisticated models producing measurable predictions, and yet you allow your human superstition and poor analytical skills to dictate the conditions of your experiments. I can't but ask "why?".

Consider a simple model, a feedforward neural network. Even this seemingly straightforward architecture exists within a vast space of design choices. To illustrate the scale of this space, its "morphology" if you will, let me discretize the key dimensions that impact model performance:

| Variable | Choices | Count |
|----------|---------|-------|
| Sequence length | 5, 10, 20, 30, 50 | 5 |
| Hidden size | 32, 64, 128, 256, 512 | 5 |
| Activation function | ReLU, Tanh, Sigmoid | 3 |
| Epochs | 10, 25, 50, 100, 200 | 5 |
| Batch size | 16, 32, 64, 128, 256 | 5 |
| Learning rate | 1e-4, 1e-3, 1e-2, 1e-1 | 4 |
| Optimizer type | Adam, SGD, RMSprop | 3 |
| Gradient clipping | None, 0.5, 1.0, 2.0 | 4 |
| Weight decay | None, 1e-4, 1e-3, 1e-2 | 4 |
| Patience | None, 5, 10, 20 | 4 |
| Model initialization | Default, Xavier, He | 3 |

Total combinations: 5 × 5 × 3 × 5 × 5 × 4 × 3 × 4 × 4 × 4 × 3 = ~864,000 possible configurations

Even for a simple feedforward network, the morphological space is massive! Any approach relying on manual experimentation to search such a vast space is at best going to be impractical - this is a problem that calls for a structured quantitative approach.
