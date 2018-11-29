#include <iostream>
#include <vector>
#include <fstream>
#include <functional>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include <string>
#include <tr1/unordered_map>
#include <Eigen/Core>
#include <random>
#include <stdio.h>
#include "utils.h"

#define RHO 0.95
#define EPSILON 0.000001
#define RATE 0.05

// 実行コマンド
// For sparse coding : make
// For non - negative sparse coding : make nonneg
// ## #Running the executable
// For sparse coding : sparse.o
// For non - negative sparse coding : nonneg.o
// Usage :./sparse.o vec_corpus factor l1_reg l2_reg num_cores outfilename
// Example :./sparse.o sample_vecs.txt 10 0.5 1e-5 1 out_vecs.txt
// This example would expand the vectors in sample_vecs.txt to 10 times their original length.

using namespace std;
using namespace Eigen;

template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

/* General parameters of the model */
template <typename T>
class Param {

 public:
  T var;

  void Init(const int& rows, const int& cols) {
    if (cols == 1) {
      // var = (0.6 / sqrt (rows)) * T::Random(rows, 1);
      var = (0.6 / sqrt(rows)) * T::Ones(rows, 1);
      // cout << "辞書\n" << rows << "\n";
      _del_var = T::Zero(rows, 1);
      _del_grad = T::Zero(rows, 1);
    }
    // var = (0.6 / sqrt (rows + cols)) * T::Random(rows, cols);
    var = (0.6 / sqrt(rows + cols)) * T::Ones(rows, cols);
    _del_var = T::Zero(rows, cols);
    _del_grad = T::Zero(rows, cols);
    _grad_sum = T::Zero(rows, cols);
    _epsilon = EPSILON * T::Ones(rows, cols);
  }

  void AdagradUpdate(const double& rate, const T& grad) {
    // printf("%d\n", grad.size()); // 900000
    // printf("%d\n", grad.rows()); // 300 : L
    // printf("%d\n", grad.cols()); // 3000 : K
    // printf("\n");
    _del_grad += grad.cwiseAbs2();
    _grad_sum += grad;
    var -= rate * grad.cwiseQuotient(_del_grad.cwiseSqrt());
  }

  void AdagradUpdateWithL1Reg(const double& rate, const T& grad,
                              const double& l1_reg) {
    // printf("%d\n", grad.size()); // 3000
    // printf("%d\n", grad.rows()); // 3000
    // printf("%d\n", grad.cols()); // 1
    // printf("\n");
    _update_num += 1;
    // cerr << '\n'
    //      << grad
    //      << '\n';
    _del_grad += grad.cwiseAbs2();
    _grad_sum += grad;
    // cerr << '\n'
    //      << _del_grad
    //      << '\n';
    double diff_pre;
    double gamma;
    for (int i = 0; i < var.rows(); ++i){    // 3000 : K
      for (int j = 0; j < var.cols(); ++j) { // 1 : 1
        double diff = abs(_grad_sum(i, j)) - _update_num * l1_reg;
        diff_pre = diff;
        gamma = -sgn(_grad_sum(i, j)) * rate * diff / sqrt(_del_grad(i, j));
        if (diff <= 0) 
          var(i, j) = 0;
        else
          var(i, j) = gamma;
      }
    }
    // cerr << '\n'
    //      << diff_pre
    //      << '\n';
  }

  void AdagradUpdateWithL1RegNonNeg(const double& rate, const T& grad,
                                    const double& l1_reg) {
    _update_num += 1;
    _del_grad += grad.cwiseAbs2();
    _grad_sum += grad;
    for (int i = 0; i < var.rows(); ++i) {
      for (int j = 0; j < var.cols(); ++j) {
        double diff = abs(_grad_sum(i, j)) - _update_num * l1_reg;
        if (diff <= 0)
          var(i, j) = 0;
        else {
          double temp = -sgn(_grad_sum(i, j)) * rate * diff /
                        sqrt(_del_grad(i, j));
          if (temp >= 0) var(i, j) = temp;
          else var(i, j) = 0;
        }
      }
    }
  }

  void WriteToFile(ofstream& out) {
    out << var.rows() << " " << var.cols() << " ";
    for (unsigned i = 0; i < var.rows(); ++i) {
      for(unsigned j = 0; j < var.cols(); ++j) 
        out << var(i, j) << " ";
    }
    out << endl;
  }

  void ReadFromFile(ifstream& in) {
    string line;
    getline(in, line);
    vector<string> data = split_line(line, ' ');
    int rows = stoi(data[0]), cols = stoi(data[1]);
    var = T::Zero(rows, cols);
    for (int i = 2; i < data.size(); ++i)
      var((i-2)/cols, (i-2)%cols) = stod(data[i]);
  }

 private:
  T _del_var, _del_grad, _grad_sum;  // updates/gradient memory
  T _epsilon;
  int _update_num = 0;
};

/* Main class definition that learns the word vectors */
class Model {

 public:
  /* The parameters of the model */
  vector<Param<Col> > atom;
  Param<Mat> dict;
  int vec_len, factor;
      
  Model(const int& times, const int& vector_len, const int& vocab_len) {
    vec_len = vector_len;
    factor = times;
    dict.Init(vec_len, factor * vec_len);
    /* Params initialization */
    for (int i = 0; i < vocab_len; ++i) {
      Param<Col> vec;
      vec.Init(factor * vec_len, 1);
      atom.push_back(vec);
    }
  }

  template<typename T> void NonLinearity(T* vec) { ElemwiseHardTanh(vec); }

  void PredictVector(const Col& word_vec, const int& word_index,
                     Col* pred_vec) {
    *pred_vec = dict.var * atom[word_index].var;
  }

  void UpdateParams(const int& word_index, const double& rate,
                    const Col& diff_vec, const double& l1_reg,
                    const double& l2_reg) {
    Mat dict_grad = -2 * diff_vec * atom[word_index].var.transpose() +
                    2 * l2_reg * dict.var;
    dict.AdagradUpdate(rate, dict_grad);
    // printf("%d\n", dict.var.transpose().rows()); // 3000 : K
    // printf("%d\n", dict.var.transpose().cols()); // 300 : L
    // printf("%d\n", diff_vec.rows());             // 300 : L
    // printf("%d\n", diff_vec.cols());             // 1
    // printf("\n");
    Col atom_elem_grad = -2 * dict.var.transpose() * diff_vec;
    // cerr << '\n'
    //      << 
    //      << '\n';
    atom[word_index].AdagradUpdateWithL1Reg(rate, atom_elem_grad, l1_reg);
  }

  void WriteVectorsToFile(const string& filename,
                          const mapUnsignedStr& vocab) {
    ofstream outfile(filename);
    if (outfile.is_open()) {
      outfile.precision(3);
      for(unsigned i = 0; i < atom.size(); ++i) {
        auto it = vocab.find(i);
        outfile << it->second << " ";
        for (unsigned j = 0; j < atom[i].var.rows(); ++j)
          outfile << atom[i].var[j] << " ";
        outfile << endl;
      }
      outfile.close();
      cerr << "\nWritten vectors to: " << filename;
    } else {
      cerr << "\nFailed to open " << filename;
    }
  }

  void WriteDictToFile(const string& filename) {
    ofstream outfile(filename);
    if (outfile.is_open()) {
      outfile.precision(3);
      dict.WriteToFile(outfile);
      outfile.close();
      cerr << "\nWritten atom to: " << filename;
    } else {
      cerr << "\nFailed to open " << filename;
    }
  }

};

void Train(const string& out_file, const int& factor,
           const int& cores, const double& l1_reg, const double& l2_reg,
           const vector<Col>& word_vecs, const mapUnsignedStr& vocab) {
  Model model(factor, word_vecs[0].size(), word_vecs.size());
  double avg_error = 1, prev_avg_err = 0;
  int iter = 0;
  while (iter < 2 && (avg_error > 0.05 && iter < 50 && abs(avg_error - prev_avg_err) > 0.005)) {
    iter += 1;
    cerr << "\nIteration: " << iter << endl;
    unsigned num_words = 0;
    double total_error = 0, atom_l1_norm = 0;
    int word_id;
    #pragma omp parallel num_threads(cores) shared(total_error,atom_l1_norm)
    #pragma omp for nowait private(word_id)
    for (int word_id = 0; word_id < word_vecs.size(); ++word_id) {
      /* Predict the i-th word and compute error */
      Col pred_vec;
      model.PredictVector(word_vecs[word_id], word_id, &pred_vec);
      Col diff_vec = word_vecs[word_id] - pred_vec;
      // cerr << "\n"
      //      << pred_vec;
      double error = diff_vec.squaredNorm();
      #pragma omp critical
      {
        total_error += error;
        num_words += 1;
        // atom_l1_norm += model.atom[word_id].var.lpNorm<1>();
        cerr << "num_words: " << num_words << "\r";
      }
      // cerr << "\n error_ "
      //      << num_words
      //      << ": "
      //      << error;
      model.UpdateParams(word_id, RATE, diff_vec, l1_reg, l2_reg);
      
      atom_l1_norm += model.atom[word_id].var.lpNorm<1>();

      // prev_avg_err = avg_error;
      // avg_error = total_error / num_words;
      // cerr << "\nError per example: " << total_error / num_words;
      // cerr << "\nDict L2 norm: " << model.dict.var.lpNorm<2>();
      // cerr << "\nAvg Atom L1 norm: " << atom_l1_norm / num_words << "\n";
      //model.WriteVectorsToFile(out_file, vocab);
    }
    prev_avg_err = avg_error;
    avg_error = total_error / num_words;
    cerr << "\nError per example: "<< total_error / num_words;
    cerr << "\nDict L2 norm: " << model.dict.var.lpNorm<2>();
    cerr << "\nAvg Atom L1 norm: " << atom_l1_norm / num_words << "\n";
    model.WriteVectorsToFile(out_file, vocab);
  }
  model.WriteVectorsToFile(out_file, vocab);
  model.WriteDictToFile(out_file + "_dict");
}

int main(int argc, char **argv) {
  mapUnsignedStr vocab;
  vector<Col> word_vecs;
  if (argc == 7) {
    string vec_corpus = argv[1];
    int factor = stoi(argv[2]);
    double l1_reg = stod(argv[3]), l2_reg = stod(argv[4]);
    int num_cores = stoi(argv[5]);
    string outfilename = argv[6];

    ReadVecsFromFile(vec_corpus, &vocab, &word_vecs);
 
    cerr << "Model specification" << endl;
    cerr << "----------------" << endl;
    cerr << "Input vector length: " << word_vecs[0].size() << endl;
    cerr << "Output vector length: " << factor * word_vecs[0].size() << endl;
    cerr << "L2 Reg (Dict): " << l2_reg << endl;
    cerr << "L1 Reg (Atom): " << l1_reg << endl;
    cerr << "Number of Cores: " << num_cores << endl;
    cerr << "----------------" << endl;

    Train(outfilename, factor, num_cores, l1_reg, l2_reg, word_vecs, vocab);
  } else {
    cerr << "Usage: "<< argv[0] << " vec_corpus factor l1_reg l2_reg "
         << "num_cores outfilename\n";
  }
  return 1;
}
