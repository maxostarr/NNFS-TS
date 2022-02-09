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
  forward(input: tf.Tensor2D): tf.Tensor2D {
    this.output = tf.maximum(input, 0);
    return this.output;
  }
}

class Activation_Softmax {
  output: tf.Tensor2D = tf.zeros([1, 1]);

  forward(input: tf.Tensor2D): tf.Tensor2D {
    const exp_values: tf.Tensor2D = input.exp().sub(input.max(1, true));
    const norm_values: tf.Tensor2D = exp_values.div(exp_values.sum(1, true));
    this.output = norm_values;
    return norm_values;
  }
}

abstract class Loss {
  calculate(output: tf.Tensor2D, y: tf.Tensor<tf.Rank.R1 | tf.Rank.R2>) {
    const sample_loss: tf.Scalar = this.forward(output, y);
    const loss: tf.Scalar = sample_loss.mean();
    return loss;
  }

  abstract forward(
    output: tf.Tensor2D,
    y: tf.Tensor<tf.Rank.R1 | tf.Rank.R2>,
  ): tf.Scalar;
}

class Loss_CategoricalCrossEntropy extends Loss {
  forward(
    y_pred: tf.Tensor2D,
    y_true: tf.Tensor<tf.Rank.R1 | tf.Rank.R2>,
  ): tf.Scalar {
    const y_pred_clipped = tf.clipByValue(y_pred, 1e-7, 1.0 - 1e-7);
    if (y_true.shape.length === 1) {
      const correct_confidences = tf.stack(
        (y_true as unknown as tf.Tensor1D).arraySync().map((v,i)=>y_pred_clipped.slice([i,v],[1,1]).squeeze())
      )
    }
    if (y_true.shape.length === 2) {
      const correct_confidences = tf.sum(y_pred_clipped.mul(y_true), 1);
    }
    return tf.scalar(0);
  }
}

const output_test = tf.tensor2d([
  [0.7, 0.1, 0.2],
  [0.1, 0.5, 0.4],
  [0.02, 0.9, 0.08],
]);

const y_test = tf.tensor1d([0, 1, 1], "int32");
const y_test_one_hot = tf.tensor2d([
  [1, 0, 0],
  [0, 1, 0],
  [0, 1, 0],
]);

const loss_test = new Loss_CategoricalCrossEntropy();
loss_test.calculate(output_test, y_test);
loss_test.calculate(output_test, y_test_one_hot);

// const POINTS = 100;
// const CLASSES = 3;

// const [X, y] = spiral_data(POINTS, CLASSES);

// const data: Plot[] = [
//   {
//     x: math
//       .subset(X, math.index(math.range(0, POINTS * CLASSES), 0))
//       .toArray()
//       .flat(),
//     y: math
//       .subset(X, math.index(math.range(0, POINTS * CLASSES), 1))
//       .toArray()
//       .flat(),
//     type: "scatter",
//     mode: "markers",
//     marker: {
//       color: y.toArray().flat(),
//     },
//   },
// ];

// const dense1 = new Layer_Dense(2, 3);
// const activation1 = new Activation_ReLU();

// const dense2 = new Layer_Dense(3, 3);
// const activation2 = new Activation_Softmax();

// dense1.forward(tf.tensor2d(X.toArray()));
// activation1.forward(dense1.output);

// dense2.forward(activation1.output);
// activation2.forward(dense2.output);

// console.log(activation2.output.print());
