import * as tf from "@tensorflow/tfjs-node";

const layer_outputs = tf.tensor2d([
  [4.8, 1.21, 2.385],
  [8.9, -1.81, 0.2],
  [1.41, 1.051, 0.026],
]);

const exp_values = layer_outputs.exp();
// sum(1, true)
// 1 to ensure it sums along the rows
// and true to keep the dimension of the sum so we get a column vector out
const norm_values = exp_values.div(exp_values.sum(1, true));
console.log(norm_values.print());

// // const E = 2.718281828459045;
// const E = Math.E;
// const exp = (x: number): number => Math.pow(E, x);

// const exp_values = layer_outputs.map((x) => exp(x));

// const normalize = (arr: number[]): number[] => {
//   const sum = arr.reduce((a, b) => a + b, 0);
//   return arr.map((x) => x / sum);
// };

// // const norm_base = exp_values.reduce((a, b) => a + b);
// // const norm_values = exp_values.map((x) => x / norm_base);

// const normalized = normalize(exp_values);
// console.log(
//   "🚀 ~ file: p006-softmax_activation.ts ~ line 18 ~ normalized",
//   normalized,
// );
