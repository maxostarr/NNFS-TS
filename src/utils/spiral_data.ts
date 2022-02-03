import * as math from "mathjs";

// Moved this code from spiral-data.js written by @vancegillies
// Updated by @daniel-kukiela
export function spiral_data(points: number, classes: number) {
  // Using MathJs functions to make matrices with zeros but converting to arrays for simplicity
  let X = math.zeros(points * classes, 2);
  let y = math.zeros(points * classes, "dense");
  let ix = 0;
  for (let class_number = 0; class_number < classes; class_number++) {
    let r = 0;
    let t = class_number * 4;

    while (r <= 1 && t <= (class_number + 1) * 4) {
      // adding some randomness to t
      const random_t = t + math.random(points) * 0.008;
      // Was `* 0.2` but reduced so you can somewhat see the arms of spiral in visualization
      // Fell free to change it back

      // converting from polar to cartesian coordinates
      // X[ix][0] = r * math.sin(random_t * 2.5);
      X = math.subset(X, math.index(ix, 0), math.sin(random_t * 2.5) * r);
      // X[ix][1] = r * math.cos(random_t * 2.5);
      X = math.subset(X, math.index(ix, 1), math.cos(random_t * 2.5) * r);
      // y[ix] = class_number;
      y = math.subset(y, math.index(ix), class_number);

      // the below two statements achieve linspace-like functionality
      r += 1.0 / (points - 1);
      t += 4.0 / (points - 1);

      ix++; // increment index
    }
  }
  // Returning as MathJs matrices, could be arrays, doesnt really matter
  return [math.matrix(X), math.matrix(y)];
}
