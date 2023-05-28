#include "cvnp/cvnp.h"
#include "libcbdetect/boards_from_corners.h"
#include "libcbdetect/config.h"
#include "libcbdetect/find_corners.h"
#include "libcbdetect/get_init_location.h"
#include "libcbdetect/image_normalization_and_gradients.h"
#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace cbdetect;
using namespace pybind11::literals;

PYBIND11_MODULE(cbdetect_py, m) {
  m.doc() = R"pbdoc(
        Corner detector
        -----------------------

        .. currentmodule:: cbdetect_py

        .. autosummary::
           :toctree: _generate

    )pbdoc";

  py::enum_<DetectMethod>(m, "DetectMethod")
      .value("TemplateMatchFast", DetectMethod::TemplateMatchFast)
      .value("TemplateMatchSlow", DetectMethod::TemplateMatchSlow)
      .value("HessianResponse", DetectMethod::HessianResponse)
      .value("LocalizedRadonTransform", DetectMethod::LocalizedRadonTransform)
      .export_values();

  py::enum_<CornerType>(m, "CornerType")
      .value("SaddlePoint", CornerType::SaddlePoint)
      .value("MonkeySaddlePoint", CornerType::MonkeySaddlePoint)
      .export_values();

  py::class_<Params>(m, "Params")
      .def(py::init<>())
      .def_readwrite("show_processing", &Params::show_processing)
      .def_readwrite("show_debug_image", &Params::show_debug_image)
      .def_readwrite("show_grow_processing", &Params::show_grow_processing)
      .def_readwrite("norm", &Params::norm)
      .def_readwrite("polynomial_fit", &Params::polynomial_fit)
      .def_readwrite("norm_half_kernel_size", &Params::norm_half_kernel_size)
      .def_readwrite("polynomial_fit_half_kernel_size",
                     &Params::polynomial_fit_half_kernel_size)
      .def_readwrite("init_loc_thr", &Params::init_loc_thr)
      .def_readwrite("score_thr", &Params::score_thr)
      .def_readwrite("strict_grow", &Params::strict_grow)
      .def_readwrite("overlay", &Params::overlay)
      .def_readwrite("occlusion", &Params::occlusion)
      .def_readwrite("detect_method", &Params::detect_method)
      .def_readwrite("corner_type", &Params::corner_type)
      .def_readwrite("radius", &Params::radius);

  py::class_<Corner>(m, "Corner")
      .def(py::init<>())
      .def_readwrite("p", &Corner::p)
      .def_readwrite("r", &Corner::r)
      .def_readwrite("v1", &Corner::v1)
      .def_readwrite("v2", &Corner::v2)
      .def_readwrite("v3", &Corner::v3)
      .def_readwrite("score", &Corner::score);

  py::class_<Board>(m, "Board")
      .def(py::init<>())
      .def_readwrite("idx", &Board::idx)
      .def_readwrite("energy", &Board::energy)
      .def_readwrite("num", &Board::num);
  m.def("find_corners", &find_corners, "Find corners in the image", "img"_a,
        "params"_a);
  m.def("boards_from_corners", &boards_from_corners,
        "Generate boards from the corners", "img"_a, "corners"_a, "params"_a);
  m.def(
      "hessian_response",
      [](const cv::Mat img) {
        cv::Mat img_resized, img_norm, img_du, img_dv, img_angle, img_weight;
        double scale = 0;
        if (img.rows < 640 || img.cols < 480) {
          scale = 2.0;
        } else {
          scale = 0.5;
        }
        cv::resize(img, img_resized,
                   cv::Size(img.cols * scale, img.rows * scale), 0, 0,
                   cv::INTER_LINEAR);
        if (img_resized.channels() == 3) {
#if CV_VERSION_MAJOR >= 4
          cv::cvtColor(img_resized, img_norm, cv::COLOR_BGR2GRAY);
#else
          cv::cvtColor(img_resized, img_norm, CV_BGR2GRAY);
#endif
          img_norm.convertTo(img_norm, CV_64F, 1 / 255.0, 0);
        } else {
          img_resized.convertTo(img_norm, CV_64F, 1 / 255.0, 0);
        }
        image_normalization_and_gradients(img_norm, img_du, img_dv, img_angle,
                                          img_weight, Params());
        cv::Mat gauss_img;
        cv::GaussianBlur(img_norm, gauss_img, cv::Size(7, 7), 1.5, 1.5);
        cv::Mat hessian_img;
        hessian_response(gauss_img, hessian_img);
        // double mn = 0, mx = 0;
        // cv::minMaxIdx(hessian_img, &mn, &mx, NULL, NULL);
        hessian_img = cv::abs(hessian_img);
        return hessian_img;
      },
      "Calculate the hessian response for the image", "img"_a);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
