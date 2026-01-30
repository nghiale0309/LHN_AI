#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "lhn/LinearRegression.h"
#include "lhn/LogisticRegression.h"

#include <lhn/physics/nn/siren_physics_net.hpp>
#include <lhn/physics/train/physics_trainer.hpp>
#include <lhn/physics/train/batch_train.hpp>
#include <lhn/physics/sampling/lensing_sampler.hpp>
#include <lhn/physics/train/train_batch_poisson.hpp>

namespace py = pybind11;
using namespace lhn::physics;

PYBIND11_MODULE(core_backend, m) {

    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<>())
        .def("fit", &LinearRegression::fit)
        .def("get_weight", &LinearRegression::get_weight)
        .def("get_bias", &LinearRegression::get_bias);

    py::class_<LogisticRegression>(m, "LogisticRegression")
        .def(py::init<double, int>(),
             py::arg("lr") = 0.1,
             py::arg("max_iter") = 1000)
        .def("fit", &LogisticRegression::fit)
        .def("predict", &LogisticRegression::predict)
        .def("predict_proba", &LogisticRegression::predict_proba)
        .def("get_weight", &LogisticRegression::get_weight)
        .def("get_bias", &LogisticRegression::get_bias);

    py::class_<nn::SirenPhysicsNet>(m, "SirenPhysicsNet")
        .def(py::init<const std::vector<int>&, double>())
        .def("forward", py::overload_cast<double, double>(&nn::SirenPhysicsNet::forward))
        .def("laplacian_batch", [](nn::SirenPhysicsNet& net, py::array_t<double> X) {
            auto r = X.unchecked<2>();
            size_t N = r.shape(0);
            std::vector<double> flat_X;
            flat_X.reserve(N * 2);
            for (size_t i = 0; i < N; i++) {
                flat_X.push_back(r(i, 0));
                flat_X.push_back(r(i, 1));
            }
            return net.laplacian_batch(flat_X);
        });

    py::class_<training::PhysicsTrainer>(m, "PhysicsTrainer")
        .def(py::init<nn::SirenPhysicsNet&, double, double, double>(),
             py::keep_alive<1, 2>())
        .def("step", py::overload_cast<double, double, double>(&training::PhysicsTrainer::step));

    py::class_<sampling::LensingSampler>(m, "LensingSampler")
        .def(py::init<>())
        .def("sample",
            [](sampling::LensingSampler& s, int region) {
                double x, y;
                s.sample(x, y, region);
                return std::vector<double>{x, y};
            }
        );

    m.def("train_batch",
        &training::train_batch,
        py::arg("trainer"),
        py::arg("sampler"),
        py::arg("batch_size"),
        py::arg("step"),
        py::arg("total_steps")
    );

    m.def("train_batch_poisson",
        [](training::PhysicsTrainer& trainer,
           py::array_t<double, py::array::c_style> X,
           py::array_t<double, py::array::c_style> kappa,
           int epochs) {
            
            try {
                auto buf_X = X.request();
                auto buf_kappa = kappa.request();

                if (buf_X.ndim != 2 || buf_X.shape[1] != 2) {
                    throw std::runtime_error("X must be shape (N, 2)");
                }
                if (buf_kappa.ndim != 1) {
                    throw std::runtime_error("kappa must be shape (N,)");
                }
                if (buf_X.shape[0] != buf_kappa.shape[0]) {
                    throw std::runtime_error("X and kappa must have same number of samples");
                }

                double* ptr_X = static_cast<double*>(buf_X.ptr);
                double* ptr_kappa = static_cast<double*>(buf_kappa.ptr);
                size_t n_samples = buf_X.shape[0];

                training::train_batch_poisson(trainer, ptr_X, ptr_kappa, n_samples, epochs);
            } catch (const std::exception& e) {
                std::cerr << "C++ Error: " << e.what() << std::endl;
                throw;
            }
        },
        py::arg("trainer"),
        py::arg("X"),
        py::arg("kappa"),
        py::arg("epochs")
    );
}