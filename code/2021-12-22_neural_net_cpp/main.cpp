//
// Created by Lyndon Duong on 12/21/21.
//

#include "matrix.h"
#include "nn.h"
#include <fstream>
#include <deque>
//#include <iomanip>

// helper to initialize multi-layer perceptron with n hidden layers each w/ same num hidden units
auto make_model(size_t in_channels,
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

void test_matrix();

auto mean(const auto &d) {
  float mu {0.};
  for(auto v: d){
    mu += v;
  }
  return mu/d.size();
}

void log(auto &file, const auto &x, const auto &y, const auto &y_hat){
  auto mse = (y.data[0] - y_hat.data[0]);
  mse = mse*mse;

  file << mse << " "
       << x.data[0] << " "
       << y.data[0] << " "
       << y_hat.data[0] << " \n";
}

int main() {
//  test_matrix();

  std::srand(42069);

  // init model
  int in_channels{1}, out_channels{1}, hidden_units_per_layer{8}, hidden_layers{3};
  float lr{.5f};
//  auto model = make_model(in_channels, out_channels, hidden_units_per_layer, hidden_layers, lr);
  auto model = make_model(
      in_channels=1,
      out_channels=1,
      hidden_units_per_layer=8,
      hidden_layers=3,
      lr=.5f);

  // train
  std::ofstream my_file;
  my_file.open ("data2.txt");
  int max_iter{1000}, print_every{500};
  float mse;
  auto deque = std::deque<float>(print_every);
  for(int i = 1; i<=max_iter; ++i) {
    // generate (x, y) training data
    auto x = lynalg::mtx<float>::rand(in_channels, 1).multiply_scalar(3.);
    auto y = x.apply_function([](float v) -> float {return sin(v)*sin(v);});

    auto y_hat = model(x);  // forward pass
    model.backprop(y); // backward pass

    // compute and print error
    mse = (y - y_hat).square().data[0];
    deque.push_back(mse);
    if ((i+1)%print_every==0) {
      log(my_file, x, y, y_hat);
//      my_file << mse << " " << x.data[0] << " " << y.data[0] << " " << y_hat.data[0] << " \n";
//      std::cout << std::setprecision(4) << std::scientific << "iter: " << i << " -- loss: " << mean(deque) << std::endl;
    }

  }
  my_file.close();

}

void test_matrix() {
  auto M = mtx<float>::randn(2, 3); // init randn matrix

  M.print_shape();
  M.print(); // print the OG matrix

  (M - M).print();  // print M minus itself

  (M + M).print();  // print its sum
  (M.multiply_scalar(2.f)).print();  // print 2x itself

  (M.multiply_elementwise(M)).print(); // mult M w itself

  auto MT = M.T(); // transpose the matrix
  MT.print();
  (MT.matmul(M)).print();  // form symmetric positive definite matrix

  (M.apply_function([](auto x) { return x - x; })).print(); // apply function

  M.print();
  M.sum().print();
  M.sum(0).print();
  M.sum(1).print();

  M.mean().print();
  M.mean(0).print();
  M.mean(1).print();

  M.cat(M, 0).print();
  M.cat(M, 1).print();

  auto uno = mtx<float>::ones(2, 2);

  uno.print();
  uno.diag().print();
  uno.diag().diag().print();
}