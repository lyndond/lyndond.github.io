//
// Created by Lyndon Duong on 12/21/21.
//

#include "matrix.h"
#include "nn.h"
#include <limits>

// helper to initialize multi-layer perceptron with n hidden layers each w/ same num hidden units
nn::MLP<float> make_model(size_t in_channels,
                          size_t out_channels,
                          size_t hidden_units_per_layer,
                          int hidden_layers,
                          float lr) {
  std::vector<size_t> units_per_layer;

  units_per_layer.push_back(in_channels);

  for (int i = 0; i < hidden_layers; ++i)
    units_per_layer.push_back(hidden_units_per_layer);

  units_per_layer.push_back(out_channels);

  nn::MLP<float> model(units_per_layer, 0.01f);
  return model;
}

// helper function to create training data
Matrix<float> linear_nonlinear(Matrix<float> x, Matrix<float> A, Matrix<float> b) {
  auto y = A.matmul(x) + b;
  y = y.apply_function(nn::sigmoid);
  return y;
}

void test_matrix(){
  auto M = mtx<float>::randn(2, 2); // init randn matrix

  M.print_shape();
  M.print(); // print the OG matrix

  (M-M).print();  // print M minus itself

  (M+M).print();  // print its sum
  (M.multiply_scalar(2.f)).print();  // print 2x itself

  (M.multiply_elementwise(M)).print(); // mult M w itself

  auto MT = M.T(); // transpose the matrix
  MT.print();
  (MT.matmul(M)).print();  // form symmetric positive definite matrix

  (M.apply_function([](auto x){return x-x;} )).print(); // apply function
}

int main() {
  test_matrix();
  // init the MLP
  auto in_channels{2}, out_channels{1}, hidden_units_per_layer{4}, hidden_layers{2};
  auto lr{.001f};
  nn::MLP<float> model = make_model(in_channels, out_channels, hidden_units_per_layer, hidden_layers, lr);

  // fixed linear transform y=f(Ax+b)
  auto A = lynalg::mtx<float>::randn(out_channels, in_channels);
  auto b = lynalg::mtx<float>::randn(out_channels, 1);

  // train
  auto i{1}, max_iter{10000};
  auto mse{std::numeric_limits<float>::infinity()};
  while (mse > 1E-8f && i <= max_iter) {
    // generate (x, y) training data
    auto x = lynalg::mtx<float>::randn(in_channels, 1);
    auto y = linear_nonlinear(x, A, b);

    // run through network
    auto y_hat = model(x);

    // backprop and change weights
    model.backprop(y);

    // compute and print error
    mse = (y - y_hat).square().data[0];
    std::cout << "iter " << i << "/" << max_iter << " | loss: " << mse << std::endl;
    ++i;
  }

}
