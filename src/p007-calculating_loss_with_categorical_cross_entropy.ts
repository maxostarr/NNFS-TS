const softmax_output = [0.7, 0.1, 0.2];

const target_output = [1, 0, 0];

const loss = -Math.log(softmax_output[0]);
console.log(loss);
