#include "boost/program_options.hpp"
#include "fmt/ranges.h"
#include "worker/executor.h"
#include <iostream>
#include <unordered_set>

static const std::unordered_set<std::string> kAllowedModes = {
    "plain_linear", "plain_greedy", "dp_greedy"};
static const std::string mode_desc =
    fmt::format("the mode of compiler, availabe with {}", kAllowedModes);

int main(int argc, char *argv[]) {
  using namespace fluidml;
  try {
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()("help,h", "print the help information")(
        "name,n", boost::program_options::value<std::string>()->required(),
        "the name of model")(
        "input,i", boost::program_options::value<std::string>()->required(),
        "the input onnx file")("mlir",
                               boost::program_options::value<std::string>(),
                               "the output mlir file with high level dialects")(
        "llvm", boost::program_options::value<std::string>(),
        "the output mlir file with the llvm dialect")(
        "json", boost::program_options::value<std::string>(),
        "the JSON file generated by intermediate results")(
        "mode,m",
        boost::program_options::value<std::string>()->composing()->notifier(
            [&](const std::string &mode) {
              if (kAllowedModes.find(mode) == kAllowedModes.end()) {
                throw boost::program_options::error(
                    fmt::format("{} is an invalid mode. Only {} are allowed.",
                                mode, kAllowedModes));
              }
            }),
        mode_desc.c_str())(
        "epoch,e", boost::program_options::value<size_t>()->default_value(1),
        "the number of epoch during evaluation");
    boost::program_options::variables_map vm;
    boost::program_options::store(
        boost::program_options::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << std::endl;
      return 0;
    }
    boost::program_options::notify(vm);
    std::string name = vm["name"].as<std::string>(),
                input = vm["input"].as<std::string>();
    std::optional<std::string> mlir = std::nullopt, llvm = std::nullopt,
                               json = std::nullopt;
    if (vm.count("mlir")) {
      mlir = vm["mlir"].as<std::string>();
    }
    if (vm.count("llvm")) {
      llvm = vm["llvm"].as<std::string>();
    }
    if (vm.count("json")) {
      json = vm["json"].as<std::string>();
    }
    size_t epoch = vm["epoch"].as<size_t>();
    std::unique_ptr<worker::Executor> executor =
        worker::Executor::MakePlainLinear(std::move(name), epoch);
    executor->Compile(input, mlir, llvm, json);
    return 0;
  } catch (const boost::program_options::required_option &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
