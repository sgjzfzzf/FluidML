#ifndef CPU_TRANSFORMERS_WORKER_FWD_H_
#define CPU_TRANSFORMERS_WORKER_FWD_H_

namespace cpu_transformers {
namespace worker {

class Builder;
class GeneralBuilder;
class KernelBuilder;
class NaiveBuilder;
class DynamicProgrammingBuilder;
class Converter;
class Evaluator;
class Lower;
class Parser;
class MemoryPlanner;
class LinearPlanner;
class GreedyPlanner;
class Runner;
class Scheduler;
class NaiveScheduler;
class DynamicProgrammingScheduler;

} // namespace worker
} // namespace cpu_transformers

#endif