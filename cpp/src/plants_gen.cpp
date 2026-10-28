// plants_gen: generators for Boiler, Three-Tank and PT2 dead-time, selected with --problem
// (see esbo/commands.hpp for the subcommands).
#include "boiler.hpp"
#include "esbo/commands.hpp"
#include "pt2.hpp"
#include "threetank.hpp"

int main(int argc, char** argv) {
    return esbo::guarded_main(argc, argv, "plants_gen", [](const std::string& cmd, const esbo::Args& a) {
        const std::string p = a.get("problem");
        if (a.has("no_dead_time")) throw std::runtime_error("--no-dead-time exists for furuta only");
        if (p == "boiler") return esbo::dispatch<esbo::boiler::Problem>(cmd, a, "plants_gen");
        if (p == "threetank_pi") return esbo::dispatch<esbo::threetank::Problem>(cmd, a, "plants_gen");
        if (p == "pt2_deadtime") return esbo::dispatch<esbo::pt2::Problem>(cmd, a, "plants_gen");
        throw std::runtime_error("plants_gen needs --problem boiler|threetank_pi|pt2_deadtime");
    });
}
