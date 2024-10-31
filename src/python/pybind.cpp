#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "structure/context/context.h"
#include "structure/flow/flow.h"
#include "structure/flow/sequence.h"
#include "structure/graph/graph.h"
#include "structure/memory/index.h"
#include "worker/builder.h"
#include "worker/converter.h"
#include "worker/executor.h"
#include "worker/lower.h"
#include "worker/parser.h"
#include "worker/planner.h"
#include "worker/runner.h"
#include <optional>
#include <string_view>

using namespace fluidml;

// TODO: rewrite python library part later
PYBIND11_MODULE(fluidml, m) {
  pybind11::class_<context::Context>(m, "Context")
      .def(pybind11::init<>())
      .def("make_plain_general_builder",
           &context::Context::MakePlainGeneralBuilder,
           pybind11::arg("function_name"))
      .def("make_dp_general_builder", &context::Context::MakeDPGeneralBuilder,
           pybind11::arg("function_name"))
      .def("make_kernel_builder", &context::Context::MakeKernelBuilder,
           pybind11::arg("function_name"))
      .def("make_plain_linear_planner",
           &context::Context::MakePlainLinearPlanner)
      .def("make_plain_greedy_planner",
           &context::Context::MakePlainGreedyPlanner)
      .def("make_dp_greedy_planner", &context::Context::MakeDPGreedyPlanner)
      .def("make_lower", &context::Context::MakeLower)
      .def("make_runner", &context::Context::MakeRunner);

  /* NOLINT(bugprone-unused-raii) */ pybind11::class_<
      flow::Flow, std::unique_ptr<flow::Flow>>(m, "Flow");

  /* NOLINT(bugprone-unused-raii) */ pybind11::class_<
      flow::Sequence, std::unique_ptr<flow::Sequence>>(m, "Sequence");

  /* NOLINT(bugprone-unused-raii) */ pybind11::class_<
      graph::Graph, std::unique_ptr<graph::Graph>>(m, "Graph");

  /* NOLINT(bugprone-unused-raii) */ pybind11::class_<
      memory::Index, std::unique_ptr<memory::Index>>(m, "Index");

  /* NOLINT(bugprone-unused-raii) */ pybind11::class_<
      worker::Builder, std::unique_ptr<worker::Builder>>(m, "Builder");

  pybind11::class_<worker::GeneralBuilder,
                   std::unique_ptr<worker::GeneralBuilder>>(m, "GeneralBuilder")
      .def("run", &worker::GeneralBuilder::Run, pybind11::arg("sequence"),
           pybind11::arg("index"));

  pybind11::class_<worker::Converter, std::unique_ptr<worker::Converter>>(
      m, "Converter")
      .def("run", &worker::Converter::Run, pybind11::arg("graph"))
      .def_static("make", &worker::Converter::Make);

  pybind11::class_<worker::Executor, std::unique_ptr<worker::Executor>>(
      m, "Executor")
      .def(
          "compile",
          pybind11::overload_cast<
              std::string_view, std::optional<std::string_view>,
              std::optional<std::string_view>, std::optional<std::string_view>>(
              &worker::Executor::Compile),
          pybind11::arg("input"), pybind11::arg("mlir") = std::nullopt,
          pybind11::arg("llvm") = std::nullopt,
          pybind11::arg("json") = std::nullopt)
      .def("invoke",
           pybind11::overload_cast<
               const std::unordered_map<std::string, pybind11::array> &>(
               &worker::Executor::Invoke),
           pybind11::arg("args"))
      .def_static("make_plain_linear", worker::Executor::MakePlainLinear,
                  pybind11::arg("name"), pybind11::arg("epoch") = 1)
      .def_static("make_plain_greedy", worker::Executor::MakePlainGreedy,
                  pybind11::arg("name"), pybind11::arg("epoch") = 1)
      .def_static("make_dp_greedy", worker::Executor::MakeDPGreedy,
                  pybind11::arg("name"), pybind11::arg("epoch") = 1);

  pybind11::class_<worker::Lower, std::unique_ptr<worker::Lower>>(m, "Lower")
      .def("run", &worker::Lower::Run);

  pybind11::class_<worker::Parser, std::unique_ptr<worker::Parser>>(m, "Parser")
      .def("run",
           pybind11::overload_cast<std::string_view>(&worker::Parser::Run),
           pybind11::arg("input"))
      .def_static("make", worker::Parser::Make);

  pybind11::class_<worker::Planner, std::unique_ptr<worker::Planner>>(m,
                                                                      "Planner")
      .def("run", &worker::Planner::Run, pybind11::arg("flow"));

  pybind11::class_<worker::Runner, std::unique_ptr<worker::Runner>>(m, "Runner")
      .def(
          "run",
          pybind11::overload_cast<
              const std::unordered_map<std::string, pybind11::array> &, size_t>(
              &worker::Runner::Run),
          pybind11::arg("args"), pybind11::arg("epoch") = 1);
}
