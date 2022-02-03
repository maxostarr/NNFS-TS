import * as tf from "@tensorflow/tfjs-node";
import * as math from "mathjs";
import { plot, Plot } from "nodeplotlib";

import { spiral_data } from "./utils/spiral_data";

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

class Activation_ReLU {
  output: tf.Tensor2D = tf.zeros([1, 1]);

  // constructor() {
  //   // this.output = tf.zeros([1, input.shape[1]]);
  // }

  forward(input: tf.Tensor2D): tf.Tensor2D {
    this.output = tf.maximum(input, 0);
    return this.output;
  }
}

const POINTS = 100;
const CLASSES = 3;

const [X, y] = spiral_data(POINTS, CLASSES);

const data: Plot[] = [
  {
    x: math
      .subset(X, math.index(math.range(0, POINTS * CLASSES), 0))
      .toArray()
      .flat(),
    y: math
      .subset(X, math.index(math.range(0, POINTS * CLASSES), 1))
      .toArray()
      .flat(),
    type: "scatter",
    mode: "markers",
    marker: {
      color: y.toArray().flat(),
    },
  },
];

const layer1 = new Layer_Dense(2, 5);
const activation1 = new Activation_ReLU();
const layer2 = new Layer_Dense(5, 2);

layer1.forward(tf.tensor2d(X.toArray()));
activation1.forward(layer1.output);
console.log(activation1.output.print());

// plot(data);
