#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "structure/context/context.h"
#include "structure/flow/flow.h"
#include "structure/flow/sequence.h"
#include "structure/memory/greedy.h"
#include "structure/memory/linear.h"
#include "structure/memory/plan.h"
#include "worker/builder.h"
#include "worker/converter.h"
#include "worker/lower.h"
#include "worker/parser.h"
#include "worker/planner.h"
#include "worker/runner.h"

PYBIND11_MODULE(libCpuTransformers, m) {

  pybind11::class_<cpu_transformers::context::Context,
                   std::shared_ptr<cpu_transformers::context::Context>>(
      m, "Context")
      .def(pybind11::init())
      .def("Make", &cpu_transformers::context::Context::Make)
#ifdef DEBUG
      .def("DumpModule", &cpu_transformers::context::Context::DumpModule)
#endif
      ;

  /* NOLINT(bugprone-unused-raii) */ pybind11::class_<
      cpu_transformers::graph::Graph>(m, "Graph");

  /* NOLINT(bugprone-unused-raii) */ pybind11::class_<
      cpu_transformers::flow::Flow>(m, "Flow");

  /* NOLINT(bugprone-unused-raii) */ pybind11::class_<
      cpu_transformers::flow::Sequence>(m, "Sequence");

  /* NOLINT(bugprone-unused-raii) */ pybind11::class_<
      cpu_transformers::memory::Index>(m, "Index");

  /* NOLINT(bugprone-unused-raii) */ pybind11::class_<
      cpu_transformers::memory::Plan>(m, "Plan");

  /* NOLINT(bugprone-unused-raii) */ pybind11::class_<
      cpu_transformers::memory::LinearPlan>(m, "LinearPlan");

  /* NOLINT(bugprone-unused-raii) */ pybind11::class_<
      cpu_transformers::memory::GreedyPlan>(m, "GreedyPlan");

  pybind11::class_<cpu_transformers::worker::NaiveBuilder>(m, "NaiveBuilder")
      .def(
          pybind11::init<std::string &&,
                         std::shared_ptr<cpu_transformers::context::Context>>())
      .def("Run", &cpu_transformers::worker::Builder::Run);

  pybind11::class_<cpu_transformers::worker::Converter>(m, "Converter")
      .def(pybind11::init())
      .def("Run", &cpu_transformers::worker::Converter::Run);

  pybind11::class_<cpu_transformers::worker::Lower>(m, "Lower")
      .def(
          pybind11::init<std::shared_ptr<cpu_transformers::context::Context>>())
      .def("Run", &cpu_transformers::worker::Lower::Run);

  pybind11::class_<cpu_transformers::worker::Parser>(m, "Parser")
      .def(pybind11::init())
      .def("Run", pybind11::overload_cast<const std::string &>(
                      &cpu_transformers::worker::Parser::Run));

  pybind11::class_<cpu_transformers::worker::LinearPlanner>(m, "LinearPlanner")
      .def(pybind11::init())
      .def("FlowToSequence", &cpu_transformers::worker::Planner::FlowToSequence)
      .def("Run", &cpu_transformers::worker::Planner::Run);

  pybind11::class_<cpu_transformers::worker::GreedyPlanner>(m, "GreedyPlanner")
      .def(pybind11::init())
      .def("FlowToSequence", &cpu_transformers::worker::Planner::FlowToSequence)
      .def("Run", &cpu_transformers::worker::Planner::Run);

  pybind11::class_<cpu_transformers::worker::Runner>(m, "Runner")
      .def(
          pybind11::init<std::shared_ptr<cpu_transformers::context::Context>>())
      .def("Run", pybind11::overload_cast<
                      const std::unordered_map<std::string, pybind11::array> &>(
                      &cpu_transformers::worker::Runner::Run));
}
