import * as tf from "@tensorflow/tfjs-node";

class Layer_Dense {
  weights: tf.Tensor2D;
  biases: tf.Tensor1D;
  output: tf.Tensor2D;

  constructor(n_inputs: number, n_neurons: number) {
    const shape: [number, number] = [n_inputs, n_neurons];
    this.weights = tf.randomUniform(shape, -0.3, 0.3, "float32", 0);
    this.biases = tf.zeros([n_neurons]);
    this.output = tf.zeros([1, n_neurons]);
  }

  forward(input: tf.Tensor2D): tf.Tensor2D {
    this.output = input.dot(this.weights).add(this.biases) as tf.Tensor2D;
    return this.output;
  }
}

const X = tf.tensor2d([
  [1, 2, 5, 2.5],
  [2.0, 5.0, -1.0, 2.0],
  [-1.5, 2.7, 3.3, -0.8],
]);

const layer1 = new Layer_Dense(4, 5);
const layer2 = new Layer_Dense(5, 2);

layer1.forward(X);
layer2.forward(layer1.output);
console.log(layer2.output.print());
