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

That is `5 × 5 × 3 × 5 × 5 × 4 × 3 × 4 × 4 × 4 × 3 = ~864,000` possible configurations before even considering the input data or its preprocessing!

Even for a simple feedforward network, the morphological space is massive! Any approach relying on manual experimentation to search such a vast space is at best going to be impractical - this is a problem that calls for a structured quantitative approach.

## Making sense of the mess
In a previous job of mine, I worked in an innovation hub at one of the largest architectural firms in Sweden. My role at the time, was as _computational designer_, an odd term that often provided people with more questions than answers. As a job title it was not ideal, but it was an interesting job. It involved applying programmatic methods for exploring and manufacturing complex three dimensional forms. The tool I used allowed for geometric optimization by means of a built in genetic algorithm solver. Genetic algorithms (GA) is an idea that takes its inspiration from Darwinian evolution theory and employs operations such as "mutation" and "crossover" on candidate solutions, hypothesizing that two fit solutions are likely to have fit offspring. As it turns out GA is also a commonly used tool for optimizing machine learning training parameters, model architectures and for feature selection.

A few weeks back and with all of the above in mind I wrote a simple event driven GA library with Postgres persistence for Rust. I will not be going in to the technical details of the library here as this chapter is about putting to the test. However, I welcome you to check it out, should you be interested:

https://github.com/jakob-lilliemarck/fx-durable-ga

The question on my mind was:
> Could genetic algorithms be a way to search for "predictive ability" over feature selection, preprocessing, model architecture and training parameters combined?

Searching for a good solution across all of those, is after all the reality of training just about any model on any dataset, and surely that's a space is bound to be so large that its impractical, improbable and perhaps even impossible, for any manual process to find a global optimum, or even a good local one. In search of an answer to my question I set up a project to attempt a combined search for predictive ability using the "Beijing Multi-Site Air Quality" dataset (Chen, 2017) while making use of my new library `fx-durable-ga`. You can find the project here:

https://github.com/jakob-lilliemarck/fx-fitness-search

### Genetic encoding and fitness
So how does one connect a genetic algorithm to a research question, and get it to search and find candidate solutions? Lets for a moment consider a much smaller and much simpler search space to illustrate this process.

Lets imagine that we're searching for a point `A` within a cube. The cube can be represented using the cartesian coordinate system, within which it occupies some space along each of the three axis `X`, `Y` and `Z`. The geometric bounds of the cube could be defined as a maximum and a minimum value along each axis. For simplicities sake, let's say the cube occupies the space between the point `{0.0, 0.0, 0.0}` and `{1.0, 1.0, 1.0}`, the full cube could then be described as:
```json
{
  x_axis: { min: 0.0, max: 1.0 },
  y_axis: { min: 0.0, max: 1.0 },
  z_axis: { min: 0.0, max: 1.0 }
}
```

In terms of our search that definition now defines all valid candidate solutions, the infinite number of points within the cube, and separates them from the much larger infinite number of invalid solutions outside of the cube. As such, the definitions effectively captures the _morphology_ of our candidate solution.

We could then say that the _fittest candidate solution_ (a point) will be at zero distance from `A`, the point we're search for.

Now we mostly have what we need to generate candidate solutions. For reasons beyond this example, the framework I've written requires genes to be integers, so in order to adapt this example and adhere to those requirements we'll need to also define the number of steps, the resolution, along each axis. In other words we'll need to _discretize_ the space.

Every valid point along an axis `{ min: 0.0, max: 1.0, steps: 11 }` could then be described as one of the 11 integer numbers between 0 and 10. We now possess the means of _encoding_ and _decoding_ a point to its "genomic representation" through the _morphological bounds_ of the cube. To give an example:
```json
// morphology
[
  { min: 0.0, max: 1.0, steps: 11 }, // x axis
  { min: 0.0, max: 1.0, steps: 11 }, // y axis
  { min: 0.0, max: 1.0, steps: 11 }  // z axis
]

// point candidate solution
point = {
  x: 0.2,
  y: 0.1,
  z: 0.9
}

// encoded as genome
encoded = [
  2, // 0.2 * (11 - 1) = 0.2 * 10 = 2, subtract 1 from 11 as both bounds are inclusive
  1, // 0.1 * (11 - 1) = 0.1 * 10 = 1
  9  // 0.9 * (11 - 1) = 0.9 * 10 = 9
]

decoded = {
  x: 0.2, // 2 / (11 - 1) = 2 / 10 = 0.2
  y: 0.1, // 1 / (11 - 1) = 1 / 10 = 0.1
  z: 0.9  // 9 / (11 - 1) = 9 / 10 = 0.9
}
```

Using the morphological definition, the GA framework and now _generate_, _breed_ and _mutate_ genomes as well as evaluating their _fitness_ as their distance from `A`. That is essentially the basics of a genetic algorithm optimization solver.

### Putting it in context

### The experiment
TBD

### Constraining the model
TBD

### Conclusions and next steps
TBD
